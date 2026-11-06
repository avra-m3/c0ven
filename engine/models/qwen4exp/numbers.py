"""The numbers of qwen4exp that the control plane needs, each with its source.

Sizes are from spec/MODEL_LAYER.md, spec/MODEL_IO.md and MEMORY_BUDGET.md; token ids from spec/MODEL_IO.md,
section 4.1, and reference/prompts.jsonl. Where a number is an estimate it says so.
"""

from __future__ import annotations

# Token ids (spec/MODEL_IO.md, section 4.1).
IM_START = 248_045
IM_END = 248_046
EOS_IDS = (IM_END, 248_044)
"""Generation stops at <|im_end|> or <|endoftext|> (generation_config.json, spec/MODEL_IO.md, section 2.3)."""

# Tokens the chat template's turns start with, read from reference/prompts.jsonl: `<|im_start|>` then the role.
SYSTEM = 8_678
ASSISTANT = 74_455
NEWLINE = 198

# Per sequence (MEMORY_BUDGET.md, "Per sequence"). K and V and the indexer's keys come from the tensor table
# (memory.py).
DELTANET_STATE_BYTES = 3_268_608
"""One DeltaNet layer's state (spec/MODEL_LAYER.md, 2.7)."""
PLE_HISTORY_BYTES = 368_640
"""The PLE conv history, which one layer holds (spec/MODEL_LAYER.md, 2.7)."""
DRAFT_KV_BYTES = 35_651_584
"""The draft layer's ring of 32,768 cells."""
RESIDENT_CELLS = 32_768
"""K and V cells kept in VRAM per sequence when KV streams (spec/STRATA_MEMORY.md, section 5)."""
