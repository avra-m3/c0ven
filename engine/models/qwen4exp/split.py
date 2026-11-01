"""How each tensor is laid over the two cards (ADR-0023; DESIGN.md, sections 5 and 9).

Each card holds half of every layer. The rules, by tensor:

- Projections into heads or channels (attention q, k, v and gate; DeltaNet qkv, gate and per-head parameters; the
  hyper-connections' weights; the shared expert's gate and up; the PLE's projections; the head's vocabulary) are
  halved by output rows: each card computes its half of the heads or values.
- Projections out of them (attention and DeltaNet output; the shared expert's down) are halved by input columns:
  each card gives a partial sum, and exchange 1 or 2 adds them.
- The router is whole on both cards: both route the token in full and pick the same experts with no third exchange
  (DESIGN.md, section 9, step 4). So are the norms and the indexer's projections, since the indexer's keys are split
  by position, not by head (DESIGN.md, section 4).
- The token embedding is halved by columns; each card holds half of every row.
- Routed experts are split by rows when cached, and stay whole in the host arena when not.
- The n-gram table is read from the file (ADR-0015).

This is the table `open` passes; the layer program in the core and its kernels follow it.
"""

from __future__ import annotations

import re

from engine.api.interface import SplitKind
from engine.models.qwen4exp.tensors import Category, FileTensor

_RULES: list[tuple[re.Pattern[str], SplitKind]] = [
    (re.compile(p), k)
    for p, k in [
        (r"blk\.\d+\.ffn_(gate|up)_exps\.weight", SplitKind.EXPERT_ROWS),
        (r"blk\.\d+\.ffn_down_exps\.weight", SplitKind.EXPERT_COLUMNS),
        (r"blk\.\d+\.ffn_gate_inp(_shexp)?\.weight", SplitKind.WHOLE),
        (r"blk\.\d+\.ffn_(gate|up)_shexp\.weight", SplitKind.ROWS),
        (r"blk\.\d+\.ffn_down_shexp\.weight", SplitKind.COLUMNS),
        (r"blk\.\d+\.(attn_output|ssm_out)\.weight", SplitKind.COLUMNS),
        (r"blk\.\d+\.(attn_q|attn_k|attn_v|attn_gate|attn_qkv)\.weight", SplitKind.ROWS),
        (r"blk\.\d+\.(attn_q_norm|attn_k_norm|ssm_norm)\.weight", SplitKind.WHOLE),
        (r"blk\.\d+\.ssm_(a|dt\.bias|alpha\.weight|beta\.weight|conv1d\.weight)", SplitKind.ROWS),
        (r"blk\.\d+\.indexer\..*", SplitKind.WHOLE),
        (r"blk\.\d+\.hc_(attn|ffn)_norm\.weight", SplitKind.WHOLE),
        (r"blk\.\d+\.hc_(attn|ffn)_(down|up|inject)\.weight", SplitKind.ROWS),
        (r"blk\.\d+\.ple_norm_.*", SplitKind.WHOLE),
        (r"blk\.\d+\.ple_(conv1d|key|value)\.weight", SplitKind.ROWS),
        (r"token_embd\.weight", SplitKind.COLUMNS),
        (r"output\.weight", SplitKind.ROWS),
        (r"output_hc_norm\.weight", SplitKind.WHOLE),
        (r"output_hc_(down|up)\.weight", SplitKind.ROWS),
        (r"per_layer_token_embd\.weight", SplitKind.FILE),
    ]
]


def split_of(name: str) -> SplitKind:
    for pattern, kind in _RULES:
        if pattern.fullmatch(name):
            return kind
    raise KeyError(f"no split rule for {name}")


def bytes_per_card(t: FileTensor, cards: int = 2) -> int:
    """What the tensor takes on each card outside the expert cache: whole, its share of the cards, or nothing."""
    if t.category is Category.EXPERTS:
        return 0
    match split_of(t.name):
        case SplitKind.WHOLE:
            return t.nbytes
        case SplitKind.ROWS | SplitKind.COLUMNS:
            return -(-t.nbytes // cards)
        case _:
            return 0
