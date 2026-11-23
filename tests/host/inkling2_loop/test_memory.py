from __future__ import annotations

import pytest

from engine.memory_plan import GIB, PlanOptions, plan
from engine.models.inkling2_loop import memory, tensors


def test_two_cards_fit_all_experts_in_vram():
    table = tensors.from_tsv()
    mm = memory.model_memory(table, cards=2)
    mp = plan(mm, PlanOptions(slots=2, context=65_536, cards=2, usable_vram=24 * GIB))
    assert mp.cards == 2
    assert mp.experts_per_layer == table.experts_per_layer
    assert mp.arena_bytes == 0
    assert mp.streamed_kv_bytes == 0


def test_one_card_spills_overflow_experts_to_host_arena():
    table = tensors.from_tsv()
    mm = memory.model_memory(table, cards=1)
    mp = plan(mm, PlanOptions(slots=1, context=32_768, cards=1, usable_vram=24 * GIB))
    assert mp.cards == 1
    assert 0 < mp.experts_per_layer < table.experts_per_layer
    assert mp.arena_bytes > 0


@pytest.mark.parametrize("cards", [2, 3, 4, 8])
def test_multi_card_plans_reduce_per_card_footprint(cards: int):
    table = tensors.from_tsv()
    mm = memory.model_memory(table, cards=cards)
    mp = plan(mm, PlanOptions(slots=2, context=131_072, cards=cards, usable_vram=24 * GIB))
    assert mp.cards == cards
    assert mp.experts_per_layer == table.experts_per_layer
