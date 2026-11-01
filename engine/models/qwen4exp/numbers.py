"""The numbers of qwen4exp that the control plane needs, each with its source.

Sizes are from spec/MODEL_LAYER.md, spec/MODEL_IO.md and MEMORY_BUDGET.md; token ids from spec/MODEL_IO.md,
section 4.1, and reference/prompts.jsonl. Where a number is an estimate it says so.
"""

from __future__ import annotations

ARCH = "qwen4exp"

LAYERS = 48
EXPERTS_PER_LAYER = 512
CONTEXT = 262_144
VOCAB = 248_320

# Token ids (spec/MODEL_IO.md, section 4.1).
ENDOFTEXT = 248_044
IM_START = 248_045
IM_END = 248_046
THINK = 248_068
THINK_END = 248_069
EOS_IDS = (IM_END, ENDOFTEXT)
"""Generation stops at <|im_end|> or <|endoftext|> (generation_config.json, spec/MODEL_IO.md, section 2.3)."""

# Tokens the chat template's turns start with, read from reference/prompts.jsonl: `<|im_start|>` then the role.
SYSTEM = 8_678
USER = 846
ASSISTANT = 74_455
NEWLINE = 198

# Per sequence, the engine's 8-bit KV (Q8_0, 32 values in 34 B; ADR-0008), 12 attention layers
# (MEMORY_BUDGET.md, "Per sequence").
KV_CELL_BYTES = 13_056
"""K and V, all 12 attention layers, per token."""
INDEXER_CELL_BYTES = 1_632
"""The indexer's keys, all 12 attention layers, per token."""
STATE_BYTES = 118_038_528
"""36 DeltaNet layers' state and the PLE conv history (spec/MODEL_LAYER.md, 2.7)."""
DRAFT_KV_BYTES = 35_651_584
"""The draft layer's ring of 32,768 cells."""
RESIDENT_CELLS = 32_768
"""K and V cells kept in VRAM per sequence when KV streams (spec/STRATA_MEMORY.md, section 5)."""

# Per card, outside the layers' weights (MEMORY_BUDGET.md, "Per card, for the tensor split").
USABLE_VRAM = 24_385_683_456
"""Measured on each B60 (docs/measurements/MEASUREMENTS.md#2026-10-11-device-memory)."""
WORKING_PER_CARD = 1_600_663_296
"""*Estimate*: working buffers per card."""

# The draft (MTP) layer's shapes (core/models/qwen4exp/layout.h).
EMBD = 2_560
HC = 4
"""Residual streams."""
HC_LR = 320
ATT_HEADS = 24
ATT_KV_HEADS = 2
ATT_DIM = 256
FF = 640
"""Expert and shared-expert width."""


def draft_per_card(cards: int) -> int:
    """The draft layer's weights on each card, as the core's load_mtp allocates them (core/models/qwen4exp/mtp.cpp):
    each card's share of the 512 experts in IQ4_NL (18 bytes per 32 values, each matrix 256-byte aligned), the
    attention heads and shared expert split as a main layer's, the rest whole in BF16 and the norms in F32."""

    def align(n: int) -> int:
        return -(-n // 256) * 256

    ff, hc_dim = FF // cards, HC * EMBD
    slot = 2 * align(ff * EMBD // 32 * 18) + align(EMBD * ff // 32 * 18)  # gate, up, down
    experts = EXPERTS_PER_LAYER * (slot + 4)  # and the slot map
    norms = 4 * (EMBD + 4 * hc_dim + 2 * ATT_DIM)
    bf16 = 2 * (
        2 * EMBD * EMBD  # fc_embedding, fc_hidden
        + 3 * (HC_LR * hc_dim + hc_dim * HC_LR)  # the two hyper-connections' and the mixer's down and up
        + 2 * HC * hc_dim  # the hyper-connections' block_inject
        + 2 * ATT_DIM * (ATT_HEADS // cards) * EMBD  # q (with its gate)
        + 2 * ATT_DIM * (ATT_KV_HEADS // cards) * EMBD  # k, v
        + EMBD * ATT_DIM * (ATT_HEADS // cards)  # o
        + (EXPERTS_PER_LAYER + 1) * EMBD  # router and shared-expert gate
        + 3 * ff * EMBD  # shared expert
    )
    return experts + norms + bf16
