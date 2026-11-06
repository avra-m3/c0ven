"""The numbers of qwen4exp that the control plane needs, each with its source.

Sizes are from spec/MODEL_LAYER.md, spec/MODEL_IO.md and MEMORY_BUDGET.md. Where a number is an estimate it says so.
Token ids are the model file's (tensors.TensorTable.token_id).
"""

from __future__ import annotations

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
