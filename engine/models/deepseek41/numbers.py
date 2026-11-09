"""The numbers of deepseek41 that the control plane needs, each with its source.

Sizes are from the GGUF's metadata and ADR-0074; token ids from the file's tokenizer. Where a number is an estimate it
says so, with its method.
"""

from __future__ import annotations

ARCH = "deepseek41"

LAYERS = 40
EXPERTS_PER_LAYER = 384
ROUTES_PER_ROW = 6
"""`expert_used_count`: the experts one row routes to in each layer."""
CONTEXT = 16_384
"""The most positions a sequence holds until the indexer's candidate filter exists (ADR-0074, item 8)."""
VOCAB = 129_280
STEP_MAX_ROWS = 8

# Token ids (the file's tokenizer.ggml.tokens).
EOS = 1
EOS_IDS = (EOS,)
USER = 128_803
ASSISTANT = 128_804

# Per sequence, on each card: the KV head and the indexer are whole on both (ADR-0074, item 2).
WINDOW_CELLS = 136
"""The window of 128 positions and one step's rows, so a step never overwrites a cell an earlier row still reads."""
WINDOW_BYTES = LAYERS * WINDOW_CELLS * 512 * 2
"""Every layer's ring of 512 f16 cells."""
COMPRESSOR_BYTES = 3 * 16 * 1024 * 4
"""16 entries of 1,024 f32 for each of the 3 sources that compress by 2."""
STATE_BYTES = WINDOW_BYTES + COMPRESSOR_BYTES
KV_CELL_BYTES = 2_560
"""Compressed rows' latents a position, in bytes of f16: 2 x 512 at the source of layers 20-39, and half that at each of
the 3 sources that keep one row for 2 positions."""
INDEXER_CELL_BYTES = 640
"""Compressed rows' index keys a position, f16: 2 x 128, by the same count of sources."""

# Per card, outside the layers' weights.
RESERVE_PER_CARD = 2**29
"""What Sysman must still show free after the fill: past it, device allocations spill to host memory."""
BUFFERS_PER_CARD = 3 * 2**28
"""*Estimate*: an allowance for working buffers with no measurement behind it, until the core's figure replaces the
whole of WORKING_PER_CARD after a first start (PlanOptions.working_per_card)."""
LISTS_PER_CARD = 51 * 2**20 * 1_060 // 960 * 6
"""*Estimate*: recorded lists, at Flash-Next's list memory a launch, for 1,060 launches at each of 6 widths. Its record
is to come with the measurements of issue #28; the core's figure replaces this after a first start."""
WORKING_PER_CARD = RESERVE_PER_CARD + BUFFERS_PER_CARD + LISTS_PER_CARD

# In host memory.
ENGRAM_CACHE_ROWS = 1 << 20
ENGRAM_ROW_BYTES = 144
HOST_HEADROOM = 5 * 2**30
"""What the plan leaves free on the host, bytes: the margin for this model's tests (issue #28). A timed run
sets its memory floor below it."""
ENGRAM_TAG_BYTES = 8 * 2**20
READ_ALIGN = 4_096
"""An `O_DIRECT` read's alignment, which each of an expert's 3 matrices may need in a staging slot."""
STAGED_LAYERS = 2
"""Layers of SSD reads the staging ring holds: the round being computed and the next."""
