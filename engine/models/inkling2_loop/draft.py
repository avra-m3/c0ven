"""DSpark drafter memory accounting from its GGUF header."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gguf

from engine.models.inkling2_loop.split import card_partition, quant_block_values

DSPARK_RING_EXTRA = 64
"""Extra KV cells past the sliding window so a speculative draft block never overwrites live keys."""
FP16_BYTES = 2

_SPLIT_BY_Q = frozenset({"attn_q.weight", "attn_output.weight"})
_SPLIT_BY_KV = frozenset({"attn_k.weight", "attn_v.weight", "attn_k_ctx.weight", "attn_v_ctx.weight"})
_SPLIT_BY_FF = frozenset({"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})


@dataclass(frozen=True, slots=True)
class DSparkInfo:
    """Sizes read from a DSpark GGUF header."""

    layers: int
    block_size: int
    head_count: int
    head_count_kv: int
    head_dim: int
    ff: int
    sliding_window: int
    swa_pattern: tuple[bool, ...]
    tensors: tuple[tuple[str, int, int], ...]  # (name, ggml_type, nbytes)

    @property
    def swa_layers(self) -> int:
        return sum(1 for swa in self.swa_pattern if swa)

    @property
    def global_layers(self) -> int:
        return sum(1 for swa in self.swa_pattern if not swa)


def _int_field(reader: gguf.GGUFReader, key: str, default: int | None = None) -> int:
    fld = reader.get_field(key)
    if fld is None:
        if default is not None:
            return default
        raise ValueError(f"DSpark GGUF missing key {key}")
    val = fld.contents()
    if isinstance(val, list):
        items = cast("list[object]", val)
        val = items[0]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(f"DSpark GGUF key {key} is not numeric: {val!r}")
    return int(val)


def read_dspark(path: Path) -> DSparkInfo:
    """Read DSpark tensor sizes and hyperparameters from its GGUF file."""
    reader = gguf.GGUFReader(path)
    arch_f = reader.get_field("general.architecture")
    arch = str(arch_f.contents()) if arch_f is not None else "dflash"
    layers = _int_field(reader, f"{arch}.block_count")
    block_size = _int_field(reader, f"{arch}.block_size", 8)
    head_count = _int_field(reader, f"{arch}.attention.head_count")
    head_count_kv = _int_field(reader, f"{arch}.attention.head_count_kv")
    head_dim = _int_field(reader, f"{arch}.attention.key_length", 128)
    ff = _int_field(reader, f"{arch}.feed_forward_length")
    window = _int_field(reader, f"{arch}.attention.sliding_window", 1024)
    pat_f = reader.get_field(f"{arch}.attention.sliding_window_pattern")
    swa_pattern = (
        tuple(bool(x) for x in cast("list[object]", pat_f.contents())) if pat_f is not None else (True,) * layers
    )
    tensors = tuple((t.name, int(t.tensor_type), int(t.n_bytes)) for t in reader.tensors)
    return DSparkInfo(layers, block_size, head_count, head_count_kv, head_dim, ff, window, swa_pattern, tensors)


def draft_weights_per_card(info: DSparkInfo, cards: int) -> int:
    """Largest card's share of DSpark GPU weights, bytes (`markov_w1.weight` stays on host for single-row lookup)."""
    kv_parts = card_partition(info.head_count_kv, cards, 1)
    ff_align = 1
    for name, gtype, _ in info.tensors:
        if name.endswith("ffn_down.weight"):
            ff_align = math.lcm(ff_align, quant_block_values(gtype))
    ff_parts = card_partition(info.ff, cards, ff_align)
    totals = [0] * cards
    for name, _, nbytes in info.tensors:
        if name == "markov_w1.weight":
            continue
        suffix = name.split(".", 2)[-1] if name.startswith("blk.") else name
        for c in range(cards):
            kv_n = kv_parts[c][1]
            ff_n = ff_parts[c][1]
            if suffix in _SPLIT_BY_Q or suffix in _SPLIT_BY_KV:
                totals[c] += nbytes * kv_n // info.head_count_kv
            elif suffix in _SPLIT_BY_FF:
                totals[c] += nbytes * ff_n // info.ff
            else:
                totals[c] += nbytes
    return max(totals)


def draft_kv_cell_per_card(info: DSparkInfo, cards: int) -> int:
    """Global-attention DSpark KV bytes per token on the largest card."""
    max_kv = max(n for _, n in card_partition(info.head_count_kv, cards, 1))
    return info.global_layers * 2 * max_kv * info.head_dim * FP16_BYTES


def draft_kv_per_card(info: DSparkInfo, cards: int) -> int:
    """Per-sequence DSpark fixed SWA FP16 KV ring on the largest card, bytes."""
    max_kv = max(n for _, n in card_partition(info.head_count_kv, cards, 1))
    ring = info.sliding_window + DSPARK_RING_EXTRA
    return info.swa_layers * ring * 2 * max_kv * info.head_dim * FP16_BYTES
