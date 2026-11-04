"""qwen4exp's numbers for the memory plan, from its tensor table and split rules (MEMORY_BUDGET.md)."""

from __future__ import annotations

from engine.api.interface import CORE_RESERVE, PAGE_CELLS, SplitKind
from engine.memory_plan import ModelMemory, PlanOptions
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.draft import Q8_BLOCK, draft_per_card, kv_per_slot, read_shapes
from engine.models.qwen4exp.split import bytes_per_card, split_of
from engine.models.qwen4exp.tensors import Category, LayerKind, TensorTable

PROMPT_CHUNK = 4352
"""The rows of a prompt chunk the core sizes its buffers for (core/api/model.h, `prompt_chunk`)."""
FP16 = 2
F32 = 4
STAGED_LAYERS = 2
"""The layers of experts not cached that the prompt path stages on each card (model.cpp, `staging`)."""
STAGE_ALIGN = 256
"""The alignment of each matrix in staging (model.cpp, `align256`)."""
WORKING_BY_CARDS = {1: 1_342_177_280, 2: 1_889_785_610}
"""*Estimate*: what the core holds on each card with no context and no KV pool, past its reserve, by the number of
cards, from the fit in MEASUREMENTS.md. The two are measured, not computed: no rule from one to the other held."""


def _stage_bytes(table: TensorTable, layer: int, cards: int) -> int:
    """What one expert of the layer takes in a card's prompt staging: its gate, up and down, each aligned."""

    def size(name: str) -> int:
        nbytes = table.by_name()[f"blk.{layer}.ffn_{name}_exps.weight"].nbytes // table.experts_per_layer // cards
        return -(-nbytes // STAGE_ALIGN) * STAGE_ALIGN

    return 2 * size("gate") + size("down")


def _widths(table: TensorTable) -> tuple[int, int]:
    """One attention layer's K and V values per token, all heads, and its indexer key's values per token: the rows
    of the first attention layer's `attn_k` and `attn_v`, and of its `indexer.k_proj`."""
    names = table.by_name()
    layer = table.layer_kinds().index(LayerKind.ATTENTION)

    def rows(name: str) -> int:
        return names[f"blk.{layer}.{name}.weight"].shape[1]

    return rows("attn_k") + rows("attn_v"), rows("indexer.k_proj")


def _q8(values: int) -> int:
    """The bytes of `values` in the 8-bit KV."""
    per, nbytes = Q8_BLOCK
    return values // per * nbytes


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
    kv_values, key_values = _widths(table)
    kv_cell = attention * _q8(kv_values)
    indexer_cell = attention * _q8(key_values)
    # The prompt path's fp16 K and V for each context token (prompt.cpp, `P.kh`, `P.vh`). Per pool token: each
    # attention layer's pooled indexer key in f32, one for each page (model.cpp, `L.pooled`), and one fp16 copy of
    # those keys split in two halves (prompt.cpp, `P.ikh`, `P.ikl`).
    fp16_kv = FP16 * kv_values
    pooled_key = F32 * key_values // PAGE_CELLS
    split_key = 2 * FP16 * key_values // PAGE_CELLS
    state = kinds.count(LayerKind.DELTANET) * numbers.DELTANET_STATE_BYTES + numbers.PLE_HISTORY_BYTES
    shapes = draft_shapes if draft_shapes is not None else read_shapes()
    draft_kv_planned = numbers.DRAFT_KV_BYTES // cards

    def working_of(o: PlanOptions, pool: int) -> int:
        """What the core holds beside the plan's rows, and the difference between its K and V and indexer keys and
        those rows: the plan counts a draft ring on each slot, and raw keys for each slot's whole context."""
        rows = min(PROMPT_CHUNK, o.context // 64 * 64)
        by_context = (rows + fp16_kv) * o.context // cards
        by_pool = (attention * pooled_key + split_key) * pool // cards
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
        staging_per_uncached=STAGED_LAYERS * max(_stage_bytes(table, layer, cards) for layer in range(table.layers)),
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
        max_context=numbers.MAX_YARN_SCALE * table.context_length,
        ngram_bytes=sum(t.nbytes for t in table.tensors if split_of(t.name) is SplitKind.FILE),
        working_of=working_of,
    )
