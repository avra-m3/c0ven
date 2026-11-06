"""The memory plan against MEMORY_BUDGET.md's tables for the tensor split."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from engine.memory_plan import (
    PlanError,
    PlanOptions,
    ngram_fits,
    plan,
    read_working,
    reserve_per_sequence,
    working_file,
    write_working,
)
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.draft import draft_per_card, read_shapes
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.tensors import Category, TensorTable, from_tsv
from tests.helpers import plan_options

TABLE = from_tsv()
MODEL = model_memory(TABLE)


def test_the_sizes_follow_the_layers():
    """MEMORY_BUDGET.md's sizes for this file. Half its layers are 6 attention and 18 DeltaNet."""
    assert (MODEL.layers, MODEL.experts) == (48, 48 * 512)
    assert (MODEL.kv_cell_host, MODEL.checkpoint_bytes) == (13_056, 118_038_528)
    half = model_memory(TensorTable(tuple(t for t in TABLE.tensors if t.layer < 24)))
    assert (half.layers, half.experts) == (24, 24 * 512)
    assert half.kv_cell_host == 6 * 1_088
    assert half.indexer_cell_per_card == 6 * 136 // 2
    assert half.checkpoint_bytes == 18 * 3_268_608 + 368_640


@pytest.mark.parametrize(
    ("context", "expected"),
    [(4_096, 106_926_080), (32_768, 317_493_248), (131_072, 397_709_312), (262_144, 504_664_064)],
)
def test_reserve_per_sequence(context: int, expected: int):
    """MEMORY_BUDGET.md, "The reserve per sequence, per card"."""
    assert reserve_per_sequence(MODEL, context) == expected


@pytest.mark.parametrize(
    ("slots", "context", "cached"),
    [(1, 32_768, 18_940), (2, 32_768, 18_629), (4, 32_768, 18_009), (1, 262_144, 18_757), (4, 262_144, 17_277)],
)
def test_experts_with_every_weight_halved(slots: int, context: int, cached: int):
    """MEMORY_BUDGET.md halves every weight outside the experts; with its figure the plan gives its counts."""
    halves = (
        TABLE.bytes_of(Category.DENSE) + TABLE.bytes_of(Category.PLE) + TABLE.bytes_of(Category.EMBEDDING)
    ) // 2 + TABLE.bytes_of(Category.HEAD) // 2
    p = plan(replace(MODEL, weights_per_card=halves), plan_options(slots=slots, context=context))
    assert p.experts_cached == cached


def test_the_split_rules_keep_some_weights_whole():
    """The router, the norms and the indexer's projections are whole on both cards, so the cards hold more than
    half the dense bytes, and fewer experts than MEMORY_BUDGET.md counts."""
    p = plan(MODEL, plan_options(slots=1, context=262_144))
    assert MODEL.weights_per_card > 2_257_086_720
    assert p.experts_cached < 18_757
    assert p.free_per_card == 24_385_683_456 - MODEL.weights_per_card - 831_015_936 - 1_600_663_296 - 504_664_064
    assert p.experts_cached == p.free_per_card // (MODEL.expert_bytes // 2)


def test_the_draft_layer_as_the_core_stores_it():
    """From the committed MTP manifest: IQ4_NL experts, 1,382,400 B a slot on each of two cards, the indexer left
    out, the norms in F32, and the rest split as a main layer's or whole."""
    shapes = read_shapes()
    assert len(shapes) == 31
    assert draft_per_card(shapes, 2) == 831_015_936
    assert draft_per_card(shapes, 1) == 1_593_527_296
    assert MODEL.draft_per_card == 831_015_936


def test_a_dflash_manifest(tmp_path: Path):
    """DFlash 2's manifest lists its tensors; its projections split, its norms and base kernels widen to F32."""
    tensors = [
        {"name": "fc.weight", "dtype": "BF16", "shape": [2560, 12800]},
        {"name": "norm.weight", "dtype": "BF16", "shape": [2560]},
        {"name": "layers.0.self_attn.q_proj.weight", "dtype": "BF16", "shape": [4096, 2560]},
        {"name": "layers.0.mlp.down_proj.weight", "dtype": "BF16", "shape": [2560, 7680]},
        {"name": "layers.0.attention_conv.base_kernel", "dtype": "BF16", "shape": [4, 2560]},
    ]
    (tmp_path / "dflash-manifest.json").write_text(json.dumps({"tensors": tensors}))
    shapes = read_shapes(tmp_path)
    assert draft_per_card(shapes, 2) == 2 * 2560 * 12800 + 4 * 2560 + 4096 * 2560 + 2560 * 7680 + 4 * 4 * 2560


def test_host_side():
    p = plan(MODEL, plan_options(slots=1, context=262_144))
    assert MODEL.expert_bytes == 2_046_400
    assert p.arena_bytes == (24_576 - p.experts_cached) * 2_046_400
    assert p.streamed_kv_bytes == 13_056 * 262_144 + 35_651_584
    assert p.pages == 65_536
    assert p.kv_resident_cells == 32_768


def test_the_checkpoints_are_pinned_at_the_cores_size():
    """A checkpoint is a slot's running state only, at any context; the KV of its positions stays in its pages
    (ADR-0033). The checkpoints are in the pinned total."""
    p = plan(MODEL, plan_options(slots=1, context=262_144, checkpoints=8))
    assert p.checkpoint_bytes == 8 * 118_038_528
    assert p.pinned_bytes == p.arena_bytes + p.streamed_kv_bytes + 8 * 118_038_528
    none = plan(MODEL, plan_options(slots=1, context=262_144, checkpoints=0))
    assert none.pinned_bytes == p.pinned_bytes - 8 * 118_038_528


@pytest.mark.parametrize("slots", [1, 2, 4, 8])
def test_the_pool_is_the_same_at_any_slot_count(slots: int):
    """The slots share the pool they are given; their count does not size it."""
    assert plan(MODEL, plan_options(slots=slots, context=262_144, kv_tokens=524_288)).pages == 131_072


def test_without_a_pool_size_every_slot_has_its_context():
    assert plan(MODEL, plan_options(slots=4, context=262_144)).pages == 4 * 65_536


def test_a_pool_too_small_for_one_sequence():
    with pytest.raises(PlanError):
        plan(MODEL, plan_options(slots=4, context=262_144, kv_tokens=262_143))


def test_the_host_holds_every_page_of_the_pool():
    """The host's K and V are the pool's pages, not each slot's whole context."""
    p = plan(MODEL, plan_options(slots=4, context=262_144, kv_tokens=524_288))
    assert p.streamed_kv_bytes == 13_056 * 524_288 + 4 * 35_651_584


def test_a_plan_over_the_hosts_memory_is_refused_with_what_fits():
    full = plan(MODEL, plan_options(slots=4, context=262_144))
    with pytest.raises(PlanError, match="--slots 3") as exc:
        plan(MODEL, plan_options(slots=4, context=262_144, host_memory=full.host_bytes - 1))
    assert "the host can give" in str(exc.value)
    assert (
        plan(MODEL, plan_options(slots=4, context=262_144, host_memory=full.host_bytes)).host_bytes == full.host_bytes
    )


def test_ram_max_is_a_limit_of_its_own():
    full = plan(MODEL, plan_options(slots=1, context=262_144))
    with pytest.raises(PlanError, match="--ram-max allows") as exc:
        plan(MODEL, plan_options(slots=1, context=262_144, ram_max=full.host_bytes - 1, host_memory=2**50))
    assert "--context 131072" in str(exc.value)


def test_a_limit_nothing_fits_names_a_smaller_file():
    with pytest.raises(PlanError, match="smaller model file"):
        plan(MODEL, plan_options(slots=1, context=4096, host_memory=2**30))


@pytest.mark.parametrize("options", [plan_options(slots=0), plan_options(context=300_000), plan_options(slots=40)])
def test_impossible_plans(options: PlanOptions):
    with pytest.raises(PlanError):
        plan(MODEL, options)


def test_one_card_holds_every_weight_and_head_whole():
    one = model_memory(TABLE, cards=1)
    two = model_memory(TABLE, cards=2)
    assert one.weights_per_card > 2 * two.weights_per_card - 2 * 1024**3
    assert one.kv_cell_per_card == 2 * two.kv_cell_per_card
    p1 = plan(one, plan_options(cards=1, slots=1, context=262_144))
    p2 = plan(two, plan_options(slots=1, context=262_144))
    assert p1.experts_cached < p2.experts_cached // 2
    assert p1.arena_bytes > p2.arena_bytes


def test_the_plan_takes_the_cards_memory():
    big = plan(MODEL, plan_options(slots=1, context=262_144, usable_vram=32 * 2**30))
    ref = plan(MODEL, plan_options(slots=1, context=262_144))
    assert ref.usable_vram == 24_385_683_456
    assert big.experts_per_layer > ref.experts_per_layer


def test_a_plan_with_no_card_size_is_refused():
    with pytest.raises(PlanError, match="no card was read"):
        plan(MODEL, PlanOptions(slots=1))


def test_a_card_too_small_is_refused_with_what_fits():
    with pytest.raises(PlanError, match="short before any expert") as exc:
        plan(MODEL, plan_options(slots=4, context=262_144, usable_vram=6 * 2**30))
    assert "Options that fit" in str(exc.value)


def test_without_the_draft_layer_the_cards_cache_more():
    assert (
        plan(MODEL, plan_options(slots=1, draft=False)).experts_cached
        > plan(MODEL, plan_options(slots=1)).experts_cached
    )


def test_the_ngram_table_in_ram_counts_in_the_host_total():
    off = plan(MODEL, plan_options(slots=1))
    on = plan(MODEL, plan_options(slots=1, ngram_in_ram=True))
    assert MODEL.ngram_bytes > 28 * 10**9
    assert on.host_bytes == off.host_bytes + MODEL.ngram_bytes


def test_the_ngram_table_moves_into_ram_only_with_room_to_spare():
    gib = 2**30
    p = plan(MODEL, plan_options(slots=1, host_memory=80 * gib))
    assert not ngram_fits(p, mem_total=90 * gib)
    roomy = plan(MODEL, plan_options(slots=1, host_memory=120 * gib))
    assert ngram_fits(roomy, mem_total=128 * gib)
    assert not ngram_fits(plan(MODEL, plan_options(slots=1)), mem_total=90 * gib)


def test_the_working_memory_is_the_models_estimate_unless_the_options_give_one():
    assert MODEL.working_per_card == numbers.WORKING_PER_CARD
    assert plan(MODEL, plan_options()).working_per_card == numbers.WORKING_PER_CARD
    given = plan(MODEL, plan_options(working_per_card=numbers.WORKING_PER_CARD + 2**30))
    assert given.working_per_card == numbers.WORKING_PER_CARD + 2**30
    assert given.free_per_card == plan(MODEL, plan_options()).free_per_card - 2**30


def test_the_cores_expert_budget_gives_its_working_memory():
    p = plan(MODEL, plan_options())
    assert p.working_from(p.free_per_card) == p.working_per_card
    assert p.working_from(p.free_per_card - 2**30) == p.working_per_card + 2**30
    # Planned again with the core's figure, the plan leaves the core's budget for experts.
    q = plan(MODEL, replace(plan_options(), working_per_card=p.working_from(p.free_per_card - 2**30)))
    assert q.free_per_card == p.free_per_card - 2**30


def test_the_working_memory_is_kept_per_model_and_options(tmp_path: Path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    a = working_file(tmp_path, model, plan_options(slots=2), "draft=None")
    assert read_working(a) is None
    write_working(a, 1_700_000_000)
    assert read_working(a) == 1_700_000_000
    assert working_file(tmp_path, model, plan_options(slots=2), "draft=None") == a
    assert working_file(tmp_path, model, plan_options(slots=4), "draft=None") != a
    assert working_file(tmp_path, model, plan_options(slots=2), "draft=/m/mtp") != a
    assert working_file(tmp_path, model, plan_options(slots=2, working_per_card=1), "draft=None") == a
    a.write_text("{")
    assert read_working(a) is None
    a.write_text('{"working_per_card": -1}')
    assert read_working(a) is None
