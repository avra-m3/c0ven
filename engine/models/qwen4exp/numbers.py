"""The numbers of qwen4exp that the control plane needs, each with its source.

Sizes are from spec/MODEL_LAYER.md, spec/MODEL_IO.md and MEMORY_BUDGET.md. Where a number is an estimate it says so.
Token ids are the model file's (tensors.TensorTable.token_id).
"""

from __future__ import annotations

MAX_YARN_SCALE = 4
"""The longest context the engine accepts, as a multiple of the file's context_length; past that the core rotates by
YaRN (ADR-0073)."""

# Per sequence (MEMORY_BUDGET.md, "Per sequence"): the sizes are the model file's (memory.py).
DRAFT_WINDOW = 32_768
"""The draft layer's ring, in cells: it attends to its last 32,768 (`--mtp-window`, spec/MODEL_IO.md, section 3)."""
RESIDENT_CELLS = 32_768
"""K and V cells kept in VRAM per sequence when KV streams (spec/STRATA_MEMORY.md, section 5)."""
