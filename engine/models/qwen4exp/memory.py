"""qwen4exp's numbers for the memory plan, from its tensor table and split rules (MEMORY_BUDGET.md)."""

from __future__ import annotations

from engine.api.interface import SplitKind
from engine.memory_plan import ModelMemory
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.draft import draft_per_card, read_shapes
from engine.models.qwen4exp.split import bytes_per_card, split_of
from engine.models.qwen4exp.tensors import Category, TensorTable


def model_memory(
    table: TensorTable, cards: int = 2, draft_shapes: dict[str, tuple[int, ...]] | None = None
) -> ModelMemory:
    """The sizes on one card or each of two: one card holds every weight and every head whole. draft_shapes: the
    drafter's tensor shapes (draft.read_shapes); None: the committed MTP layer's."""
    if cards not in (1, 2):
        raise ValueError(f"the engine runs on one or two cards, not {cards}")
    expert_total = table.bytes_of(Category.EXPERTS)
    experts = numbers.LAYERS * numbers.EXPERTS_PER_LAYER
    return ModelMemory(
        cards=cards,
        weights_per_card=sum(bytes_per_card(t, cards) for t in table.tensors),
        draft_per_card=draft_per_card(draft_shapes if draft_shapes is not None else read_shapes(), cards),
        working_per_card=numbers.WORKING_PER_CARD,
        layers=numbers.LAYERS,
        experts=experts,
        expert_bytes=expert_total // experts,
        kv_cell_per_card=numbers.KV_CELL_BYTES // cards,
        indexer_cell_per_card=numbers.INDEXER_CELL_BYTES // cards,
        state_per_card=numbers.STATE_BYTES // cards,
        draft_kv_per_card=numbers.DRAFT_KV_BYTES // cards,
        resident_cells=numbers.RESIDENT_CELLS,
        kv_cell_host=numbers.KV_CELL_BYTES,
        draft_kv_host=numbers.DRAFT_KV_BYTES,
        checkpoint_bytes=numbers.STATE_BYTES,
        max_context=numbers.CONTEXT,
        ngram_bytes=sum(t.nbytes for t in table.tensors if split_of(t.name) is SplitKind.FILE),
        reference_vram=numbers.USABLE_VRAM,
    )
