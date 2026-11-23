"""General n-card tensor split rules for inkling2_loop (1 to ENGINE_MAX_CARDS cards)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from engine.api import abi
from engine.api.interface import SplitKind
from engine.models.inkling2_loop.tensors import Category, FileTensor, TensorTable

# Values per quantisation block, matching shared/quant.h.
QUANT_BLOCK_VALUES: dict[int, int] = {
    0: 1,
    1: 1,
    2: 32,
    6: 32,
    7: 32,
    8: 32,
    9: 32,
    10: 256,
    11: 256,
    12: 256,
    13: 256,
    14: 256,
    15: 256,
    16: 256,
    17: 256,
    18: 256,
    19: 256,
    20: 32,
    21: 256,
    22: 256,
    23: 256,
    29: 256,
    30: 1,
    42: 64,
}

# Bytes per quantisation block, matching shared/quant.h.
QUANT_BLOCK_BYTES: dict[int, int] = {
    0: 4,
    1: 2,
    2: 18,
    6: 22,
    7: 24,
    8: 34,
    9: 36,
    10: 84,
    11: 110,
    12: 144,
    13: 176,
    14: 210,
    15: 292,
    16: 66,
    17: 74,
    18: 98,
    19: 50,
    20: 18,
    21: 110,
    22: 82,
    23: 136,
    29: 56,
    30: 2,
    42: 18,
}

_ROW_BY_Q = frozenset(
    {
        "attn_q.weight",
        "attn_gate.weight",
        "attn_rel.weight",
        "attn_sink_proj.weight",
        "attn_q_clip.weight",
        "attn_rel_clip.weight",
    }
)
_ROW_BY_KV = frozenset(
    {
        "attn_k.weight",
        "attn_v.weight",
        "attn_k_conv.weight",
        "attn_v_conv.weight",
        "attn_k_conv_proj.weight",
        "attn_v_conv_proj.weight",
    }
)


def quant_block_values(ggml_type: int) -> int:
    return QUANT_BLOCK_VALUES.get(ggml_type, 1)


def quant_block_bytes(ggml_type: int) -> int:
    return QUANT_BLOCK_BYTES.get(ggml_type, 0)


def card_partition(total: int, cards: int, align: int = 1) -> tuple[tuple[int, int], ...]:
    """Split `total` into `cards` contiguous slices `(start, count)`, each a positive multiple of `align`."""
    if cards < 1 or cards > abi.MAX_CARDS:
        raise ValueError(f"cards must be in 1..{abi.MAX_CARDS}, got {cards}")
    if align < 1 or total % align != 0:
        raise ValueError(f"total {total} is not a multiple of alignment {align}")
    units = total // align
    if units < cards:
        raise ValueError(f"total {total} ({units} blocks of {align}) cannot split across {cards} cards")
    base, rem = divmod(units, cards)
    out: list[tuple[int, int]] = []
    pos = 0
    for c in range(cards):
        n = (base + (1 if c < rem else 0)) * align
        out.append((pos, n))
        pos += n
    return tuple(out)


@dataclass(frozen=True, slots=True)
class CardSlice:
    """One card's slice of heads, expert intermediate width and unpadded vocabulary."""

    kv0: int
    kv_n: int
    q0: int
    q_n: int
    ff0: int
    ff_n: int
    v0: int
    v_n: int


def ff_alignment(table: TensorTable) -> int:
    """Least common multiple of quant block sizes across all `ffn_down_exps.weight` tensors."""
    align = 1
    for t in table.tensors:
        if t.suffix == "ffn_down_exps.weight":
            align = math.lcm(align, quant_block_values(t.ggml_type))
    return align


def card_slices(table: TensorTable, cards: int) -> tuple[CardSlice, ...]:
    """Each card's head, expert width and vocabulary slice for `cards` cards."""
    kv_parts = card_partition(table.head_count_kv, cards, 1)
    gqa = table.head_count // table.head_count_kv
    ff_parts = card_partition(table.expert_ff, cards, ff_alignment(table))
    v_parts = card_partition(table.vocab_unpadded, cards, 1)
    return tuple(
        CardSlice(
            kv0=kv0,
            kv_n=kv_n,
            q0=kv0 * gqa,
            q_n=kv_n * gqa,
            ff0=ff0,
            ff_n=ff_n,
            v0=v0,
            v_n=v_n,
        )
        for (kv0, kv_n), (ff0, ff_n), (v0, v_n) in zip(kv_parts, ff_parts, v_parts, strict=True)
    )


def split_of(tensor: FileTensor) -> SplitKind:
    """How `tensor` is distributed across the cards."""
    if tensor.category is Category.EMBEDDING:
        return SplitKind.FILE
    if tensor.name == "output.weight":
        return SplitKind.ROWS
    s = tensor.suffix
    if s in _ROW_BY_Q or s in _ROW_BY_KV:
        return SplitKind.ROWS
    if s == "attn_output.weight":
        return SplitKind.COLUMNS
    if s == "ffn_gate_up_exps.weight":
        return SplitKind.EXPERT_ROWS
    if s == "ffn_down_exps.weight":
        return SplitKind.EXPERT_COLUMNS
    return SplitKind.WHOLE


def _align256(n: int) -> int:
    return -(-n // 256) * 256


def _matrix_bytes(rows: int, cols: int, ggml_type: int) -> int:
    bv = quant_block_values(ggml_type)
    bb = quant_block_bytes(ggml_type)
    return rows * (cols // bv) * bb


def bytes_per_card(table: TensorTable, cards: int) -> tuple[int, ...]:
    """Dense GPU weights (excluding `token_embd.weight` and routed experts) held on each card, bytes."""
    slices = card_slices(table, cards)
    totals = [0] * cards
    n_head = table.head_count
    n_kv = table.head_count_kv
    for t in table.tensors:
        if t.category in (Category.EMBEDDING, Category.EXPERTS):
            continue
        s = t.suffix
        for c, sl in enumerate(slices):
            if t.name == "output.weight":
                totals[c] += _matrix_bytes(sl.v_n, t.shape[0], t.ggml_type)
            elif s in _ROW_BY_Q:
                totals[c] += t.nbytes * sl.q_n // n_head
            elif s in _ROW_BY_KV:
                totals[c] += t.nbytes * sl.kv_n // n_kv
            elif s == "attn_output.weight":
                totals[c] += t.nbytes * sl.q_n // n_head
            else:
                totals[c] += t.nbytes
    return tuple(totals)


def expert_bytes_per_card(table: TensorTable, cards: int) -> tuple[tuple[int, ...], ...]:
    """Per card, per physical layer: bytes of one routed expert's gate+up and down slices (256-byte aligned)."""
    slices = card_slices(table, cards)
    by_name = table.by_name()
    embd = table.embedding_length
    out: list[tuple[int, ...]] = []
    for sl in slices:
        per_layer: list[int] = []
        for layer_idx in range(table.layers):
            gu = by_name[f"blk.{layer_idx}.ffn_gate_up_exps.weight"]
            dn = by_name[f"blk.{layer_idx}.ffn_down_exps.weight"]
            gate_bytes = _align256(_matrix_bytes(sl.ff_n, embd, gu.ggml_type))
            down_bytes = _align256(_matrix_bytes(embd, sl.ff_n, dn.ggml_type))
            per_layer.append(2 * gate_bytes + down_bytes)
        out.append(tuple(per_layer))
    return tuple(out)


def layer_expert_bytes(table: TensorTable) -> tuple[int, ...]:
    """Whole bytes of one routed expert in each physical layer."""
    by_name = table.by_name()
    n_exp = table.experts_per_layer
    return tuple(
        (
            by_name[f"blk.{layer_idx}.ffn_gate_up_exps.weight"].nbytes
            + by_name[f"blk.{layer_idx}.ffn_down_exps.weight"].nbytes
        )
        // n_exp
        for layer_idx in range(table.layers)
    )
