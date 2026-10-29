"""What the engine process needs from qwen4exp: its open config and its checkpoint rule.

The engine process finds this module by the GGUF's `general.architecture`, so code outside models/ never names the
model (ADR-0024, item 3).
"""

from __future__ import annotations

import os
from pathlib import Path

from engine.api.interface import OpenConfig, Tensor
from engine.memory_plan import MemoryPlan, PlanOptions, plan
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.split import split_of
from engine.models.qwen4exp.tensors import TensorTable, from_gguf_cached, from_tsv
from engine.models.qwen4exp.turns import checkpoint_positions

__all__ = ["checkpoint_positions", "draft_dir", "memory_plan", "open_config"]

CPU_WORKERS = 16
"""One per core of the 7945HX (DESIGN.md, section 9)."""
NGRAM_CACHE_ROWS = 1 << 20
TIMING_RECORDS = 1 << 16


def memory_plan(table: TensorTable, options: PlanOptions) -> MemoryPlan:
    return plan(model_memory(table), options)


def draft_dir(model: Path) -> Path | None:
    """The drafter's directory beside the model file: `dflash/` with its manifest (MODEL_FILES.md, "The DFlash 2
    drafter"), else `mtp/` with its manifest (MODEL_FILES.md, "The MTP layer"), else None."""
    for name, manifest in (("dflash", "dflash-manifest.json"), ("mtp", "mtp-manifest.json")):
        d = model.parent / name
        if (d / manifest).is_file():
            return d
    return None


def open_config(
    model: Path | None,
    options: PlanOptions,
    deterministic: bool = False,
    draft: Path | None = None,
    table_cache: Path | None = None,
) -> tuple[OpenConfig, MemoryPlan]:
    """The config for `open`. Without a model file the tensor table is the committed IQ3_S one and no tensors are
    passed: enough for the mock and the reference. draft: the draft layer's directory, or None for no drafting.
    table_cache: a directory keeping the model's tensor table between starts, or None."""
    table = from_gguf_cached(model, table_cache) if model is not None else from_tsv()
    mp = memory_plan(table, options)
    tensors = (
        tuple(
            Tensor(t.name, t.ggml_type, t.shard - 1, t.offset, t.nbytes, t.shape, split_of(t.name), t.layer)
            for t in table.tensors
        )
        if model is not None
        else ()
    )
    config = OpenConfig(
        shard_paths=tuple(str(p) for p in table.shard_paths),
        tensors=tensors,
        slots=options.slots,
        max_context=options.context,
        pages=mp.pages,
        checkpoints=options.checkpoints,
        kv_resident_cells=mp.kv_resident_cells if options.kv_stream else 0,
        experts_cached=mp.experts_cached,
        arena_bytes=mp.arena_bytes,
        working_bytes=mp.working_per_card,
        cpu_workers=int(os.environ.get("ENGINE_CPU_WORKERS", CPU_WORKERS)),
        ngram_cache_rows=NGRAM_CACHE_ROWS,
        timing_records=TIMING_RECORDS,
        eos_ids=numbers.EOS_IDS,
        vocab_size=numbers.VOCAB,
        deterministic=deterministic,
        draft_dir=str(draft) if draft is not None else "",
    )
    return config, mp
