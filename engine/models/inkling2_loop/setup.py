"""`engine.models.ModelSetup` for inkling2_loop."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from engine.api import abi
from engine.api.interface import OpenConfig, Tensor
from engine.memory_plan import MemoryPlan, PlanOptions, plan
from engine.models.inkling2_loop.memory import model_memory
from engine.models.inkling2_loop.split import split_of
from engine.models.inkling2_loop.tensors import TensorTable, from_gguf_cached, from_tsv
from engine.models.inkling2_loop.turns import ChatTokens
from engine.models.inkling2_loop.turns import checkpoint_rule as rule_of

if TYPE_CHECKING:
    from engine.scheduler import CheckpointRule

__all__ = ["checkpoint_rule", "draft_dir", "memory_plan", "open_config", "vision_file"]

_TABLES: dict[Path | None, TensorTable] = {}


def _table(model: Path | None, cache: Path | None = None) -> TensorTable:
    if model is None:
        return from_tsv()
    key = model.resolve()
    table = _TABLES.get(key)
    if table is None:
        table = from_gguf_cached(model, cache)
        _TABLES[key] = table
    return table


def _pick_gguf(directory: Path, prefix: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix == ".gguf" and p.name.lower().startswith(prefix)
    )
    if len(matches) == 1:
        return matches[0]
    inkling = [p for p in matches if "inkling" in p.name.lower()]
    return inkling[0] if len(inkling) == 1 else None


def draft_dir(model: Path) -> Path | None:
    """Locate a sibling DSpark GGUF (`dspark-*.gguf`) beside `model`."""
    return _pick_gguf(model.parent, "dspark-")


def vision_file(model: Path) -> Path | None:
    """Locate a sibling multimodal projector GGUF (`mmproj-*.gguf`) beside `model`."""
    return _pick_gguf(model.parent, "mmproj-")


def _effective_cards(table: TensorTable, requested: int) -> int:
    return max(1, min(requested, table.head_count_kv, abi.MAX_CARDS))


def memory_plan(
    model: Path | None,
    options: PlanOptions,
    draft: Path | None = None,
    table_cache: Path | None = None,
) -> MemoryPlan:
    table = _table(model, table_cache)
    cards = _effective_cards(table, options.cards)
    vision = vision_file(model) if model is not None else None
    o = replace(options, cards=cards, draft=draft is not None and options.draft)
    mm = model_memory(table, cards, draft=draft if o.draft else None, vision=vision)
    return plan(mm, o)


def open_config(
    model: Path | None,
    options: PlanOptions,
    deterministic: bool = False,
    draft: Path | None = None,
    table_cache: Path | None = None,
    *,
    cpu_workers: int = 1,
) -> tuple[OpenConfig, MemoryPlan]:
    table = _table(model, table_cache)
    mp = memory_plan(model, options, draft, table_cache)
    tensors = tuple(
        Tensor(
            name=t.name,
            ggml_type=t.ggml_type,
            shard=t.shard,
            offset=t.data_offset,
            nbytes=t.nbytes,
            shape=t.shape,
            split=split_of(t),
            layer=t.layer if t.layer is not None else -1,
        )
        for t in table.tensors
    )
    eos = (table.token_id("<|content_model_end_sampling|>"),)
    config = OpenConfig(
        shard_paths=tuple(str(p) for p in table.shard_paths),
        tensors=tensors,
        slots=mp.options.slots,
        max_context=mp.options.context,
        pages=mp.pages,
        checkpoints=mp.options.checkpoints,
        kv_resident_cells=0,
        experts_cached=mp.experts_per_layer,
        cpu_workers=cpu_workers,
        eos_ids=eos,
        vocab_size=table.vocab_unpadded,
        deterministic=deterministic,
        draft_dir=str(draft) if draft is not None and mp.options.draft else "",
        max_cards=mp.cards,
    )
    return config, mp


def checkpoint_rule(model: Path | None, table_cache: Path | None = None) -> CheckpointRule:
    return rule_of(ChatTokens.of(_table(model, table_cache)))
