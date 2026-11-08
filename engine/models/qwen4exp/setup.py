"""What the engine process needs from qwen4exp (engine.models.ModelSetup): its open config, where its drafter and
image encoder are, and its checkpoint rule.

The engine process finds this module by the GGUF's `general.architecture`, so code outside models/ never names the
model (ADR-0024, item 3).
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from engine.api.interface import OpenConfig, Tensor
from engine.memory_plan import MemoryPlan, PlanOptions, plan
from engine.models.qwen4exp.draft import DFLASH_FILES, read_shapes
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.split import split_of
from engine.models.qwen4exp.tensors import TensorTable, from_gguf_cached, from_tsv
from engine.models.qwen4exp.turns import ChatTokens
from engine.models.qwen4exp.turns import checkpoint_rule as rule_of

if TYPE_CHECKING:
    from engine.scheduler import CheckpointRule

__all__ = ["checkpoint_rule", "draft_dir", "memory_plan", "open_config", "vision_file"]

CPU_WORKERS = 16
"""One per core of the test machine's 7945HX (DESIGN.md, section 9): the count when the host is not read (the mock and
the reference)."""


def memory_plan(table: TensorTable, options: PlanOptions, draft: Path | None = None) -> MemoryPlan:
    """draft: the drafter's directory, whose manifest sizes it; None: the committed MTP layer's."""
    return plan(model_memory(table, options.cards, read_shapes(draft) if draft is not None else None), options)


@lru_cache(maxsize=4)
def table_of(model: Path | None, table_cache: Path | None) -> TensorTable:
    """The model's tensor table, read once for `open_config` and `checkpoint_rule`; without a model file, the
    committed IQ3_S one."""
    return from_gguf_cached(model, table_cache) if model is not None else from_tsv()


def checkpoint_rule(model: Path | None, table_cache: Path | None = None) -> CheckpointRule:
    """Where to checkpoint a prompt, by the model file's chat tokens (turns.py)."""
    return rule_of(ChatTokens.of(table_of(model, table_cache)))


def draft_dir(model: Path) -> Path | None:
    """The drafter's directory beside the model file: `dflash/` with DFlash 2's `model.safetensors` and `config.json`
    (MODEL_FILES.md, "The DFlash 2 drafter"), else `mtp/` with its manifest (MODEL_FILES.md, "The MTP layer"), else
    None."""
    dflash = model.parent / "dflash"
    if all((dflash / name).is_file() for name in DFLASH_FILES):
        return dflash
    mtp = model.parent / "mtp"
    return mtp if (mtp / "mtp-manifest.json").is_file() else None


def vision_file(model: Path) -> Path | None:
    """The image encoder beside the model file: the one `mmproj-*.gguf` there (MODEL_FILES.md), else None."""
    found = sorted(model.parent.glob("mmproj-*.gguf"))
    return found[0] if len(found) == 1 else None


def open_config(
    model: Path | None,
    options: PlanOptions,
    deterministic: bool = False,
    draft: Path | None = None,
    table_cache: Path | None = None,
    *,
    cpu_workers: int = CPU_WORKERS,
) -> tuple[OpenConfig, MemoryPlan]:
    """The config for `open`. Without a model file the tensor table is the committed IQ3_S one and no tensors are
    passed: enough for the mock and the reference. draft: the draft layer's directory, or None for no drafting.
    table_cache: a directory keeping the model's tensor table between starts, or None. cpu_workers: the CPU's expert
    workers, which `ENGINE_CPU_WORKERS` overrides."""
    table = table_of(model, table_cache)
    if model is not None:
        options = replace(options, draft=draft is not None)
    mp = memory_plan(table, options, draft)
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
        eos_ids=table.eos_ids,
        vocab_size=table.vocab,
        deterministic=deterministic,
        draft_dir=str(draft) if draft is not None else "",
        ngram_in_ram=options.ngram_in_ram,
    )
    return config, mp
