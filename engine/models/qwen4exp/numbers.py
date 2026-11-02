"""The numbers of qwen4exp that the control plane needs, each with its source.

Sizes are from spec/MODEL_LAYER.md, spec/MODEL_IO.md and MEMORY_BUDGET.md; token ids from spec/MODEL_IO.md,
section 4.1, and reference/prompts.jsonl. Where a number is an estimate it says so.
"""

from __future__ import annotations

ARCH = "qwen4exp"

CONTEXT = 262_144

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

# Per sequence, the engine's 8-bit KV (Q8_0, 32 values in 34 B; ADR-0008; MEMORY_BUDGET.md, "Per sequence").
KV_CELL_BYTES_PER_LAYER = 1_088
"""K and V of one attention layer, per token."""
INDEXER_CELL_BYTES_PER_LAYER = 136
"""The indexer's keys of one attention layer, per token."""
DELTANET_STATE_BYTES = 3_268_608
"""One DeltaNet layer's state (spec/MODEL_LAYER.md, 2.7)."""
PLE_HISTORY_BYTES = 368_640
"""The PLE conv history, which one layer holds (spec/MODEL_LAYER.md, 2.7)."""
DRAFT_KV_BYTES = 35_651_584
"""The draft layer's ring of 32,768 cells."""
RESIDENT_CELLS = 32_768
"""K and V cells kept in VRAM per sequence when KV streams (spec/STRATA_MEMORY.md, section 5)."""

# Per card, outside the layers' weights (MEMORY_BUDGET.md, "Per card, for the tensor split").
USABLE_VRAM = 24_385_683_456
"""Measured on each B60 (docs/measurements/MEASUREMENTS.md#2026-10-11-device-memory)."""
WORKING_PER_CARD = 1_600_663_296
"""*Estimate*: working buffers per card."""
