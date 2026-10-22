"""qwen4exp's numbers for the memory plan, from its tensor table and split rules (MEMORY_BUDGET.md)."""

from __future__ import annotations

from engine.memory_plan import ModelMemory
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.split import bytes_per_card
from engine.models.qwen4exp.tensors import Category, TensorTable


def model_memory(table: TensorTable) -> ModelMemory:
    expert_total = table.bytes_of(Category.EXPERTS)
    experts = numbers.LAYERS * numbers.EXPERTS_PER_LAYER
    return ModelMemory(
        weights_per_card=sum(bytes_per_card(t) for t in table.tensors),
        draft_per_card=numbers.DRAFT_PER_CARD,
        experts=experts,
        expert_bytes=expert_total // experts,
        kv_cell_per_card=numbers.KV_CELL_BYTES // 2,
        indexer_cell_per_card=numbers.INDEXER_CELL_BYTES // 2,
        state_per_card=numbers.STATE_BYTES // 2,
        draft_kv_per_card=numbers.DRAFT_KV_BYTES // 2,
        resident_cells=numbers.RESIDENT_CELLS,
        kv_cell_host=numbers.KV_CELL_BYTES,
        draft_kv_host=numbers.DRAFT_KV_BYTES,
        checkpoint_bytes=numbers.STATE_BYTES,
        max_context=numbers.CONTEXT,
    )
