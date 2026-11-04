"""qwen4exp's numbers for the memory plan, from its tensor table and split rules (MEMORY_BUDGET.md)."""

from __future__ import annotations

from engine.api.interface import CORE_RESERVE, SplitKind
from engine.memory_plan import ModelMemory, PlanOptions
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.draft import draft_per_card, kv_per_slot, read_shapes
from engine.models.qwen4exp.split import bytes_per_card, split_of
from engine.models.qwen4exp.tensors import Category, LayerKind, TensorTable

PROMPT_CHUNK = 4352
"""The rows of a prompt chunk the core sizes its buffers for (core/api/model.h, `prompt_chunk`)."""
FP16_KV_PER_TOKEN = 2048
"""The prompt path's fp16 K and V per context token, all cards (prompt.cpp, `P.kh`, `P.vh`)."""
POOLED_KEY_PER_TOKEN = 128
"""One attention layer's pooled indexer key per pool token, all cards (model.cpp, `L.pooled`)."""
SPLIT_KEY_PER_TOKEN = 128
"""The pooled keys split into fp16 for the prompt path, per pool token, all cards (prompt.cpp, `P.ikh`, `P.ikl`)."""
WORKING_BY_CARDS = {1: 1_342_177_280, 2: 1_889_785_610}
"""*Estimate*: what the core holds on each card with no context and no KV pool, past its reserve, by the number of
cards, from the fit in MEASUREMENTS.md. The two are measured, not computed: no rule from one to the other held."""


def model_memory(
    table: TensorTable, cards: int = 2, draft_shapes: dict[str, tuple[int, ...]] | None = None
) -> ModelMemory:
    """The sizes on one card or each of two: one card holds every weight and every head whole. draft_shapes: the
    drafter's tensor shapes (draft.read_shapes); None: the committed MTP layer's."""
    if cards not in (1, 2):
        raise ValueError(f"the engine runs on one or two cards, not {cards}")
    expert_total = table.bytes_of(Category.EXPERTS)
    experts = table.layers * table.experts_per_layer
    kinds = table.layer_kinds()
    attention = kinds.count(LayerKind.ATTENTION)
    kv_cell = attention * numbers.KV_CELL_BYTES_PER_LAYER
    indexer_cell = attention * numbers.INDEXER_CELL_BYTES_PER_LAYER
    state = kinds.count(LayerKind.DELTANET) * numbers.DELTANET_STATE_BYTES + numbers.PLE_HISTORY_BYTES
    shapes = draft_shapes if draft_shapes is not None else read_shapes()
    draft_kv_planned = numbers.DRAFT_KV_BYTES // cards

    def working_of(o: PlanOptions, pool: int) -> int:
        """What the core holds beside the plan's rows, and the difference between its K and V and indexer keys and
        those rows: the plan counts a draft ring on each slot, and raw keys for each slot's whole context."""
        rows = min(PROMPT_CHUNK, o.context // 64 * 64)
        by_context = (rows + FP16_KV_PER_TOKEN) * o.context // cards
        by_pool = (attention * POOLED_KEY_PER_TOKEN + SPLIT_KEY_PER_TOKEN) * pool // cards
        drafter = o.slots * (kv_per_slot(shapes, o.context, cards) if o.draft else 0)
        return (
            by_context
            + by_pool
            + drafter
            - o.slots * draft_kv_planned
            - indexer_cell // cards * (o.slots * o.context - pool)
        )

    return ModelMemory(
        cards=cards,
        weights_per_card=sum(bytes_per_card(t, cards) for t in table.tensors),
        draft_per_card=draft_per_card(shapes, cards),
        working_per_card=CORE_RESERVE + WORKING_BY_CARDS[cards],
        layers=table.layers,
        experts=experts,
        expert_bytes=expert_total // experts,
        kv_cell_per_card=kv_cell // cards,
        indexer_cell_per_card=indexer_cell // cards,
        state_per_card=state // cards,
        draft_kv_per_card=numbers.DRAFT_KV_BYTES // cards,
        resident_cells=numbers.RESIDENT_CELLS,
        kv_cell_host=kv_cell,
        draft_kv_host=numbers.DRAFT_KV_BYTES,
        checkpoint_bytes=state,
        max_context=numbers.CONTEXT,
        ngram_bytes=sum(t.nbytes for t in table.tensors if split_of(t.name) is SplitKind.FILE),
        working_of=working_of,
    )
