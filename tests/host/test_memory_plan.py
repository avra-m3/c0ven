"""The memory plan against MEMORY_BUDGET.md's tables for the tensor split."""

from __future__ import annotations

from dataclasses import replace

import pytest

from engine.memory_plan import PlanError, PlanOptions, plan, reserve_per_sequence
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.tensors import Category, from_tsv

TABLE = from_tsv()
MODEL = model_memory(TABLE)


@pytest.mark.parametrize(
    ("context", "expected"),
    [(4_096, 106_926_080), (32_768, 317_493_248), (131_072, 397_709_312), (262_144, 504_664_064)],
)
def test_reserve_per_sequence(context: int, expected: int):
    """MEMORY_BUDGET.md, "The reserve per sequence, per card"."""
    assert reserve_per_sequence(MODEL, context) == expected


@pytest.mark.parametrize(
    ("slots", "context", "cached"),
    [(1, 32_768, 19_151), (2, 32_768, 18_841), (4, 32_768, 18_220), (1, 262_144, 18_968), (4, 262_144, 17_489)],
)
def test_experts_with_every_weight_halved(slots: int, context: int, cached: int):
    """MEMORY_BUDGET.md halves every weight outside the experts; with its figure the plan gives its counts."""
    halves = (
        TABLE.bytes_of(Category.DENSE) + TABLE.bytes_of(Category.PLE) + TABLE.bytes_of(Category.EMBEDDING)
    ) // 2 + TABLE.bytes_of(Category.HEAD) // 2
    p = plan(replace(MODEL, weights_per_card=halves), PlanOptions(slots=slots, context=context))
    assert p.experts_cached == cached


def test_the_split_rules_keep_some_weights_whole():
    """The router, the norms and the indexer's projections are whole on both cards, so the cards hold more than
    half the dense bytes, and fewer experts than MEMORY_BUDGET.md counts."""
    p = plan(MODEL, PlanOptions(slots=1, context=262_144))
    assert MODEL.weights_per_card > 2_257_086_720
    assert p.experts_cached < 18_968
    assert p.free_per_card == 24_385_683_456 - MODEL.weights_per_card - 614_446_848 - 1_600_663_296 - 504_664_064
    assert p.experts_cached == p.free_per_card // (MODEL.expert_bytes // 2)


def test_host_side():
    p = plan(MODEL, PlanOptions(slots=1, context=262_144))
    assert MODEL.expert_bytes == 2_046_400
    assert p.arena_bytes == (24_576 - p.experts_cached) * 2_046_400
    assert p.streamed_kv_bytes == 13_056 * 262_144 + 35_651_584
    assert p.pages == 65_536
    assert p.kv_resident_cells == 32_768


def test_the_checkpoints_are_pinned_at_the_cores_size():
    """A checkpoint is a slot's running state only, at any context; the KV of its positions stays in its pages
    (ADR-0033). The checkpoints are in the pinned total."""
    p = plan(MODEL, PlanOptions(slots=1, context=262_144, checkpoints=8))
    assert p.checkpoint_bytes == 8 * 118_038_528
    assert p.pinned_bytes == p.arena_bytes + p.streamed_kv_bytes + 8 * 118_038_528
    none = plan(MODEL, PlanOptions(slots=1, context=262_144, checkpoints=0))
    assert none.pinned_bytes == p.pinned_bytes - 8 * 118_038_528


@pytest.mark.parametrize("slots", [1, 2, 4, 8])
def test_the_pool_is_the_same_at_any_slot_count(slots: int):
    """The slots share the pool they are given; their count does not size it."""
    assert plan(MODEL, PlanOptions(slots=slots, context=262_144, kv_tokens=524_288)).pages == 131_072


def test_without_a_pool_size_every_slot_has_its_context():
    assert plan(MODEL, PlanOptions(slots=4, context=262_144)).pages == 4 * 65_536


def test_a_pool_too_small_for_one_sequence():
    with pytest.raises(PlanError):
        plan(MODEL, PlanOptions(slots=4, context=262_144, kv_tokens=262_143))


def test_the_host_holds_every_page_of_the_pool():
    """The host's K and V are the pool's pages, not each slot's whole context."""
    p = plan(MODEL, PlanOptions(slots=4, context=262_144, kv_tokens=524_288))
    assert p.streamed_kv_bytes == 13_056 * 524_288 + 4 * 35_651_584


def test_a_plan_over_the_hosts_memory_is_refused_with_what_fits():
    full = plan(MODEL, PlanOptions(slots=4, context=262_144))
    with pytest.raises(PlanError, match="--slots 3") as exc:
        plan(MODEL, PlanOptions(slots=4, context=262_144, host_memory=full.host_bytes - 1))
    assert "the host can give" in str(exc.value)
    assert plan(MODEL, PlanOptions(slots=4, context=262_144, host_memory=full.host_bytes)).host_bytes == full.host_bytes


def test_ram_max_is_a_limit_of_its_own():
    full = plan(MODEL, PlanOptions(slots=1, context=262_144))
    with pytest.raises(PlanError, match="--ram-max allows") as exc:
        plan(MODEL, PlanOptions(slots=1, context=262_144, ram_max=full.host_bytes - 1, host_memory=2**50))
    assert "--context 131072" in str(exc.value)


def test_a_limit_nothing_fits_names_a_smaller_file():
    with pytest.raises(PlanError, match="smaller model file"):
        plan(MODEL, PlanOptions(slots=1, context=4096, host_memory=2**30))


@pytest.mark.parametrize("options", [PlanOptions(slots=0), PlanOptions(context=300_000), PlanOptions(slots=40)])
def test_impossible_plans(options: PlanOptions):
    with pytest.raises(PlanError):
        plan(MODEL, options)
