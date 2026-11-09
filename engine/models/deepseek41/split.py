"""How each tensor is laid over the two cards (ADR-0023 as ADR-0074, item 2, changes it). Each card runs every layer.

`SplitKind` names the direction of a cut; where the cut is unequal, the core's slot classes hold the sizes. This is the
table `open` passes; the layer program in the core and its kernels follow it.
"""

from __future__ import annotations

import re

from engine.api.interface import SplitKind
from engine.models.deepseek41.tensors import Category, FileTensor

_RULES: list[tuple[re.Pattern[str], SplitKind]] = [
    (re.compile(p), k)
    for p, k in [
        # A cached expert has part A, 1,280 gate and up rows and 5 of down's 9 blocks a row, on one card, and part B,
        # the rest, on the other, as its slot pair sets. Experts stay whole in the arena and on the SSD.
        (r"blk\.\d+\.ffn_(gate|up)_exps\.weight", SplitKind.EXPERT_ROWS),
        (r"blk\.\d+\.ffn_down_exps\.weight", SplitKind.EXPERT_COLUMNS),
        (r"blk\.\d+\.ffn_(gate|up)_shexp\.weight", SplitKind.ROWS),
        (r"blk\.\d+\.ffn_down_shexp\.weight", SplitKind.COLUMNS),
        # Both cards route in full and pick the same experts, with no exchange.
        (r"blk\.\d+\.ffn_(gate_inp|norm)\.weight", SplitKind.WHOLE),
        (r"blk\.\d+\.exp_probs_b(_vl)?\.bias", SplitKind.WHOLE),
        # Half the query heads and half the output groups; `attn_output_b` gives partial sums that an exchange adds.
        (r"blk\.\d+\.attn_(q_b|output_a)\.weight", SplitKind.ROWS),
        (r"blk\.\d+\.attn_output_b\.weight", SplitKind.COLUMNS),
        (r"blk\.\d+\.attn_(q_a|q_a_norm|kv|kv_a_norm|norm|sinks)\.weight", SplitKind.WHOLE),
        (r"blk\.\d+\.attn_compressor_(kv|gate|norm)\.weight", SplitKind.WHOLE),
        # Whole, its own `attn_q_b` too: both cards score every position and pick the same top-k.
        (r"blk\.\d+\.indexer\.(attn_q_b|attn_k|proj|k_norm)\.weight", SplitKind.WHOLE),
        (r"blk\.\d+\.hc_(attn|ffn)_(base|fn|scale)\.weight", SplitKind.WHOLE),
        # Half the value rows, and the gates of 2 of the 4 streams.
        (r"blk\.\d+\.engram_(kv|gate)\.weight", SplitKind.ROWS),
        (r"output\.weight", SplitKind.ROWS),
        (r"output_norm\.weight", SplitKind.WHOLE),
        # `token_embd` is mapped and its rows read each step; the Engram table is read from the file (ADR-0015).
        (r"(per_layer_)?token_embd\.weight", SplitKind.FILE),
    ]
]
_SHARED = re.compile(r"blk\.\d+\.ffn_(gate|up|down)_shexp\.weight")
"""Q5_K's 256-value blocks cut only at 1,280/1,024, and the larger part is on the card that does not post the layer's
misses."""


def split_of(name: str) -> SplitKind:
    """How the tensor called `name` in the file is laid over the cards. Raises KeyError for a name with no rule."""
    for pattern, kind in _RULES:
        if pattern.fullmatch(name):
            return kind
    raise KeyError(f"no split rule for {name}")


def posting_card(layer: int) -> int:
    """The card, 0 or 1, that posts the misses of `layer`, waits for them and adds the CPU's answer."""
    return layer % 2


def bytes_on_card(t: FileTensor, card: int) -> int:
    """The bytes tensor `t` takes on `card`, 0 or 1, outside the expert cache: 0 for experts and for what stays in the
    file. Raises KeyError as `split_of` does."""
    if t.category is Category.EXPERTS:
        return 0
    match split_of(t.name):
        case SplitKind.WHOLE:
            return t.nbytes
        case SplitKind.ROWS | SplitKind.COLUMNS:
            if _SHARED.fullmatch(t.name):
                return t.nbytes * (4 if card == posting_card(t.layer) else 5) // 9
            return t.nbytes // 2
        case _:
            return 0
