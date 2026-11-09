"""The memory plan against MEMORY_BUDGET.md's tables for the tensor split."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from engine.api.interface import CORE_RESERVE
from engine.api.mock import CARD_MEMORY
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
from engine.models.qwen4exp.draft import draft_per_card, read_shapes
from engine.models.qwen4exp.memory import (
    deltanet_state_bytes,
    model_memory,
    ple_history_bytes,
)
from engine.models.qwen4exp.tensors import TSV_UD_IQ4_XS, Category, TensorTable, from_tsv
from tests.helpers import plan_options

TABLE = from_tsv()
MODEL = model_memory(TABLE)


def test_no_context_given_is_the_models_whole_context():
    p = plan(MODEL, plan_options(slots=1, context=0))
    assert MODEL.trained_context == TABLE.context_length
    assert MODEL.max_context == 4 * TABLE.context_length
    assert p.options.context == MODEL.trained_context
    assert p == plan(MODEL, plan_options(slots=1, context=MODEL.trained_context))


def test_the_sizes_per_sequence_are_the_files():
    """MEMORY_BUDGET.md, "Per sequence", from the file's fields; the UD-IQ4_XS file's are the same."""
    for table in (TABLE, from_tsv(TSV_UD_IQ4_XS)):
        assert model_memory(table).kv_cell_host == 12 * 1_088
        assert deltanet_state_bytes(table) == 3_268_608
        assert ple_history_bytes(table) == 368_640
    assert MODEL.draft_kv_host == 32_768 * 1_088


def test_a_dflash_drafter_counts_the_mtp_layers_ring():
    """DFlash 2's manifest has no MTP layer: its ring is counted as the MTP layer's, as before."""
    dflash = {"layers.0.self_attn.k_proj.weight": (1024, 2560), "fc.weight": (2560, 12800)}
    assert model_memory(TABLE, draft_shapes=dflash).draft_kv_host == MODEL.draft_kv_host


def test_the_sizes_follow_the_layers():
    """MEMORY_BUDGET.md's sizes for this file. Half its layers are 6 attention and 18 DeltaNet."""
    assert (MODEL.layers, MODEL.experts) == (48, 48 * 512)
    assert (MODEL.kv_cell_host, MODEL.checkpoint_bytes) == (13_056, 118_038_528)
    half = model_memory(TensorTable(tuple(t for t in TABLE.tensors if t.layer < 24), fields=TABLE.fields))
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
    """MEMORY_BUDGET.md halves every weight outside the experts; with its figures the plan gives its counts."""
    halves = (
        TABLE.bytes_of(Category.DENSE) + TABLE.bytes_of(Category.PLE) + TABLE.bytes_of(Category.EMBEDDING)
    ) // 2 + TABLE.bytes_of(Category.HEAD) // 2
    model = replace(
        MODEL, weights_per_card=halves, working_per_card=1_600_663_296, working_of=None, staging_per_uncached=0
    )
    p = plan(model, plan_options(slots=slots, context=context))
    assert p.experts_cached == cached


def test_the_split_rules_keep_some_weights_whole():
    """The router, the norms and the indexer's projections are whole on both cards, so the cards hold more than
    half the dense bytes, and fewer experts than MEMORY_BUDGET.md counts."""
    p = plan(MODEL, plan_options(slots=1, context=262_144))
    assert MODEL.weights_per_card > 2_257_086_720
    assert p.experts_cached < 18_757
    assert p.free_per_card == 24_385_683_456 - MODEL.weights_per_card - 831_015_936 - p.working_per_card - 504_664_064
    assert p.experts_cached // 48 * (MODEL.expert_bytes // 2) * 48 + p.prompt_staging_per_card <= p.free_per_card


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


def test_past_the_trained_context_the_working_memory_grows():
    """The prompt's scores and fp16 K and V, and the pooled keys, follow the context past 262,144 tokens."""
    trained = plan(MODEL, plan_options(slots=1, context=262_144))
    long = plan(MODEL, plan_options(slots=1, context=1_048_576))
    assert long.working_per_card > trained.working_per_card
    assert long.experts_cached < trained.experts_cached


def test_the_core_s_working_figure_is_taken_as_it_is_past_the_trained_context():
    p = plan(MODEL, plan_options(slots=1, context=1_048_576, working_per_card=2_000_000_000))
    assert p.working_per_card == 2_000_000_000


def test_a_limit_nothing_fits_names_a_smaller_file():
    with pytest.raises(PlanError, match="smaller model file"):
        plan(MODEL, plan_options(slots=1, context=4096, host_memory=2**30))


@pytest.mark.parametrize("options", [plan_options(slots=0), plan_options(context=1_048_580), plan_options(slots=40)])
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
    estimate = plan(MODEL, plan_options()).working_per_card
    assert estimate > MODEL.working_per_card
    given = plan(MODEL, plan_options(working_per_card=estimate + 2**30))
    assert given.working_per_card == estimate + 2**30
    assert given.free_per_card == plan(MODEL, plan_options()).free_per_card - 2**30


MEASURED_RESERVE = 512 * 2**20
"""The core's reserve when the figures below were measured, on c0ven 212daad and 05cab3e, with cards of CARD_MEMORY."""
DFLASH_SHAPES = {f"layers.{n}.self_attn.k_proj.weight": (1024, 2560) for n in range(5)}


@pytest.mark.parametrize(
    ("cards", "slots", "context", "kv_tokens", "drafter", "core_gib"),
    [
        (2, 1, 262_144, 0, "mtp", 3.37),
        (2, 4, 262_144, 524_288, "mtp", 3.48),
        (2, 1, 65_536, 0, "mtp", 2.53),
        (2, 4, 65_536, 0, "mtp", 2.70),
        (2, 1, 262_144, 0, "dflash", 3.28),
        (2, 4, 262_144, 524_288, "dflash", 3.12),
        (2, 1, 262_144, 0, None, 3.23),
        (2, 2, 131_072, 524_288, None, 3.21),
        (2, 1, 4_096, 0, None, 2.29),
        (1, 1, 65_536, 0, None, 2.22),
        (1, 1, 262_144, 0, "mtp", 3.95),
        (1, 2, 65_536, 0, "mtp", 2.40),
    ],
)
def test_the_working_memory_estimate_is_what_the_core_reported(
    *, cards: int, slots: int, context: int, kv_tokens: int, drafter: str | None, core_gib: float
):
    """The core's figure from a load of each set of options, taken with its reserve at 512 MiB. 0.04 GiB is about 1
    expert a layer."""
    model = model_memory(TABLE, cards, DFLASH_SHAPES if drafter == "dflash" else None)
    options = plan_options(cards=cards, slots=slots, context=context, kv_tokens=kv_tokens, draft=drafter is not None)
    planned = plan(model, options).working_per_card - (CORE_RESERVE - MEASURED_RESERVE)
    assert planned / 2**30 == pytest.approx(core_gib, abs=0.04)


@pytest.mark.parametrize(
    ("cards", "slots", "context", "draft", "core_experts"),
    [
        (2, 1, 262_144, True, 337),
        (1, 1, 65_536, False, 151),
        (1, 1, 262_144, True, 110),
        (1, 2, 65_536, True, 124),
    ],
)
def test_the_plans_experts_are_the_cores_before_it_lends_kv_memory(
    *, cards: int, slots: int, context: int, draft: bool, core_experts: int
):
    """The core's "at least" count a layer, from a load of each set of options, with its 2 layers of staging. The
    card's memory is the size the core read then, past its reserve then."""
    options = plan_options(
        cards=cards,
        slots=slots,
        context=context,
        draft=draft,
        usable_vram=CARD_MEMORY + CORE_RESERVE - MEASURED_RESERVE,
    )
    assert plan(model_memory(TABLE, cards), options).experts_cached // 48 == pytest.approx(core_experts, abs=2)


def test_the_working_memory_estimate_does_not_depend_on_the_card_or_the_model_file():
    """Loads at --card-memory 20 and of UD-IQ4_XS report the same figure as IQ3_S on the whole card."""
    p = plan(MODEL, plan_options())
    assert plan(MODEL, plan_options(usable_vram=20 * 2**30)).working_per_card == p.working_per_card
    assert plan(model_memory(from_tsv(TSV_UD_IQ4_XS)), plan_options()).working_per_card == p.working_per_card


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


def test_the_arena_in_reserved_huge_pages_leaves_the_host_total():
    gib = 2**30
    off = plan(MODEL, plan_options(slots=1))
    assert off.arena_bytes > 4 * gib
    held = plan(MODEL, plan_options(slots=1, hugetlb_pool=4 * gib + gib // 2))
    assert held.arena_reserved_bytes == 4 * gib
    assert held.host_bytes == off.host_bytes - 4 * gib
    whole = plan(MODEL, plan_options(slots=1, hugetlb_pool=off.arena_bytes + gib))
    assert whole.arena_reserved_bytes == off.arena_bytes
    assert plan(MODEL, plan_options(slots=1, hugetlb_pool=0)).host_bytes == off.host_bytes
