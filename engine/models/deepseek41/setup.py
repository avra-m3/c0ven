"""What the engine process needs from deepseek41 (engine.models.ModelSetup): its open config and its checkpoint rule.
It has no drafter and no image encoder (ADR-0080).

The engine process finds this module by the GGUF's `general.architecture`, so code outside models/ never names the
model (ADR-0024, item 3).
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from engine.api.interface import OpenConfig, Tensor
from engine.memory_plan import MemoryPlan, PlanError, PlanOptions
from engine.models.deepseek41 import numbers
from engine.models.deepseek41.memory import model_memory, slot_pair_plan
from engine.models.deepseek41.split import split_of
from engine.models.deepseek41.tensors import TensorTable, from_gguf_cached, from_tsv
from engine.models.deepseek41.turns import checkpoint_positions

if TYPE_CHECKING:
    from engine.scheduler import CheckpointRule

__all__ = ["checkpoint_rule", "draft_dir", "memory_plan", "open_config", "vision_file"]

CPU_WORKERS = 16
"""One per core of the test machine's 7945HX: the count when the host is not read (the mock and the reference)."""
CORE_HAS_MODEL = False
"""Whether the core runs deepseek41 (issue #28). Until it does, a model file stops here with that reason, and the mock
and the plan from the committed table still work."""


def memory_plan(table: TensorTable, options: PlanOptions) -> MemoryPlan:
    """The plan at the model's own limits: no draft layer, at most its context, and the Engram table on the SSD whatever
    `ngram_in_ram` says (ADR-0080, item 7). table: the file's tensors. options: the plan's options. Raises PlanError
    when the options do not fit."""
    limited = replace(options, draft=False, ngram_in_ram=False, context=min(options.context, numbers.CONTEXT))
    return slot_pair_plan(model_memory(table, options.cards), limited)


def checkpoint_rule(model: Path | None, table_cache: Path | None = None) -> CheckpointRule:
    """Where to checkpoint a prompt (turns.py). The chat token ids are numbers.py's: the committed table has no
    tokenizer fields to read them from. model and table_cache: as `open_config`'s, unused."""
    del model, table_cache
    return checkpoint_positions


def draft_dir(model: Path) -> Path | None:
    """None: deepseek41 does not draft."""
    del model
    return None


def vision_file(model: Path) -> Path | None:
    """None: deepseek41 reads no images, with or without an mmproj beside the file."""
    del model
    return None


def open_config(
    model: Path | None,
    options: PlanOptions,
    deterministic: bool = False,
    draft: Path | None = None,
    table_cache: Path | None = None,
    *,
    cpu_workers: int = CPU_WORKERS,
) -> tuple[OpenConfig, MemoryPlan]:
    """The config for `open`. Without a model file the tensor table is the committed one and no tensors are passed:
    enough for the mock and the reference. draft: ignored, as deepseek41 does not draft. table_cache: a directory
    keeping the model's tensor table between starts, or None. cpu_workers: the CPU's expert workers, which
    `ENGINE_CPU_WORKERS` overrides. Raises PlanError for a model file while the core does not run the model, or when
    the options do not fit."""
    del draft
    if model is not None and not CORE_HAS_MODEL:
        raise PlanError(f"the engine core does not run {numbers.ARCH} yet")
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
        max_context=mp.options.context,
        pages=mp.pages,
        checkpoints=options.checkpoints,
        kv_resident_cells=mp.kv_resident_cells,
        experts_cached=mp.experts_cached,
        cpu_workers=int(os.environ.get("ENGINE_CPU_WORKERS", cpu_workers)),
        eos_ids=numbers.EOS_IDS,
        vocab_size=numbers.VOCAB,
        deterministic=deterministic,
    )
    return config, mp
