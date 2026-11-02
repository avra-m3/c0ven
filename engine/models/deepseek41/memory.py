"""deepseek41's numbers for the memory plan, from its tensor table and split rules (ADR-0062). The pinned step buffers
and answers are in PlanOptions.host_working: the plan has no figure of their own."""

from __future__ import annotations

from dataclasses import replace

from engine.memory_plan import MemoryPlan, ModelMemory, PlanError, PlanOptions, plan
from engine.models.deepseek41 import numbers
from engine.models.deepseek41.split import bytes_on_card
from engine.models.deepseek41.tensors import Category, TensorTable


def staging_ring_bytes(table: TensorTable) -> int:
    """The SSD reads' ring at the widest step, in bytes: every row's experts missing, in the largest layer's slots.
    table: the file's tensors."""
    experts = min(numbers.ROUTES_PER_ROW * numbers.STEP_MAX_ROWS, numbers.EXPERTS_PER_LAYER)
    slot = max(table.expert_bytes(n) for n in range(numbers.LAYERS)) + 3 * numbers.READ_ALIGN
    return numbers.STAGED_LAYERS * experts * slot


def model_memory(table: TensorTable, cards: int = 2) -> ModelMemory:
    """The sizes on each of two cards. table: the file's tensors. cards: the cards in use. Raises PlanError for another
    count: one card cannot hold the dense weights and enough experts within the host's memory (ADR-0062)."""
    if cards != 2:
        raise PlanError(f"deepseek41 runs on two cards, not {cards}")
    weights = tuple(sum(bytes_on_card(t, card) for t in table.tensors) for card in range(cards))
    experts = numbers.LAYERS * numbers.EXPERTS_PER_LAYER
    return ModelMemory(
        cards=cards,
        weights_per_card=max(weights),
        draft_per_card=0,
        working_per_card=numbers.WORKING_PER_CARD,
        layers=numbers.LAYERS,
        experts=experts,
        expert_bytes=table.bytes_of(Category.EXPERTS) // experts,
        kv_cell_per_card=numbers.KV_CELL_BYTES,
        indexer_cell_per_card=numbers.INDEXER_CELL_BYTES,
        state_per_card=numbers.STATE_BYTES,
        draft_kv_per_card=0,
        resident_cells=numbers.CONTEXT,
        kv_cell_host=0,
        draft_kv_host=0,
        checkpoint_bytes=numbers.STATE_BYTES,
        max_context=numbers.CONTEXT,
        ngram_bytes=table.bytes_of(Category.NGRAM),
        reference_vram=numbers.USABLE_VRAM,
        card_weights=weights,
        layer_expert_bytes=tuple(table.expert_bytes(n) for n in range(numbers.LAYERS)),
        ssd_tier=True,
        host_headroom=numbers.HOST_HEADROOM,
        engram_cache=numbers.ENGRAM_CACHE_ROWS * numbers.ENGRAM_ROW_BYTES + numbers.ENGRAM_TAG_BYTES,
        ssd_staging_ring=staging_ring_bytes(table),
        routes_per_row=numbers.ROUTES_PER_ROW,
    )


def slot_pair_plan(m: ModelMemory, options: PlanOptions) -> MemoryPlan:
    """The plan with every layer's experts counted in slot pairs, so each tier's count is even. m: the model's sizes.
    options: the plan's options. Raises PlanError as `plan` does.

    A layer's cache has as many part-A slots as part-B slots (ADR-0062, item 2), and a pair of them takes exactly one
    expert's bytes on each card, as the parts are 5/9 and 4/9 of it. The plan's fill carries what a layer leaves over
    to the next, so counting the arena in pairs too loses no bytes there."""

    def twice(counts: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(2 * n for n in counts)

    pairs = replace(
        m,
        experts=m.experts // 2,
        expert_bytes=2 * m.expert_bytes,
        layer_expert_bytes=twice(m.layer_expert_bytes),
    )
    p = plan(pairs, options)
    if p.tiers is None:
        raise PlanError("deepseek41 has an SSD tier")
    tiers = replace(
        p.tiers,
        layer_expert_bytes=m.layer_expert_bytes,
        vram=twice(p.tiers.vram),
        arena=twice(p.tiers.arena),
        ssd=twice(p.tiers.ssd),
    )
    return replace(
        p,
        experts_cached=2 * p.experts_cached,
        experts_total=m.experts,
        card_layer_experts=tuple(twice(c) for c in p.card_layer_experts),
        tiers=tiers,
    )
