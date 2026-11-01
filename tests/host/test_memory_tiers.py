"""The memory plan's tiers: a model with an SSD tier plans VRAM, arena and SSD as sizes, unequal cards are exact, and
Flash-Next's plan and report are held to literal values."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine import fit
from engine.api.interface import EngineFacts
from engine.host import Card, Host
from engine.memory_plan import GIB, SSD_TIER_HEADROOM, ModelMemory, PlanError, PlanOptions, plan
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.tensors import from_tsv
from engine.process import report

MIB = 2**20
TABLE = from_tsv()

# qwen4exp's plan for each set of options, as literals: (options, the plan's fields). A change here changes Flash-Next.
GOLDEN = [
    (
        PlanOptions(),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 504664064,
            "reserve_per_card": 504664064,
            "free_per_card": 19107479552,
            "experts_cached": 18674,
            "experts_total": 24576,
            "arena_bytes": 12077852800,
            "streamed_kv_bytes": 3458203648,
            "checkpoint_bytes": 944308224,
            "pages": 65536,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=1, context=32768),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 317493248,
            "reserve_per_card": 317493248,
            "free_per_card": 19294650368,
            "experts_cached": 18857,
            "experts_total": 24576,
            "arena_bytes": 11703361600,
            "streamed_kv_bytes": 463470592,
            "checkpoint_bytes": 944308224,
            "pages": 8192,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=4, context=32768),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 317493248,
            "reserve_per_card": 1269972992,
            "free_per_card": 18342170624,
            "experts_cached": 17926,
            "experts_total": 24576,
            "arena_bytes": 13608560000,
            "streamed_kv_bytes": 1853882368,
            "checkpoint_bytes": 944308224,
            "pages": 32768,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=4, context=262144),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 504664064,
            "reserve_per_card": 2018656256,
            "free_per_card": 17593487360,
            "experts_cached": 17194,
            "experts_total": 24576,
            "arena_bytes": 15106524800,
            "streamed_kv_bytes": 13832814592,
            "checkpoint_bytes": 944308224,
            "pages": 262144,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=2, context=65536, draft=False),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 0,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 344231936,
            "reserve_per_card": 688463872,
            "free_per_card": 19754695680,
            "experts_cached": 19306,
            "experts_total": 24576,
            "arena_bytes": 10784528000,
            "streamed_kv_bytes": 1782579200,
            "checkpoint_bytes": 944308224,
            "pages": 32768,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=4, context=32768, kv_tokens=262144),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 317493248,
            "reserve_per_card": 1269972992,
            "free_per_card": 18342170624,
            "experts_cached": 17926,
            "experts_total": 24576,
            "arena_bytes": 13608560000,
            "streamed_kv_bytes": 3565158400,
            "checkpoint_bytes": 944308224,
            "pages": 65536,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=1, context=32768, cards=1),
        {
            "cards": 1,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 4514173440,
            "draft_per_card": 1593527296,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 634986496,
            "reserve_per_card": 634986496,
            "free_per_card": 16042332928,
            "experts_cached": 7839,
            "experts_total": 24576,
            "arena_bytes": 34250596800,
            "streamed_kv_bytes": 463470592,
            "checkpoint_bytes": 944308224,
            "pages": 8192,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=1, context=262144, usable_vram=34359738368),
        {
            "cards": 2,
            "usable_vram": 34359738368,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 504664064,
            "reserve_per_card": 504664064,
            "free_per_card": 29081534464,
            "experts_cached": 24576,
            "experts_total": 24576,
            "arena_bytes": 0,
            "streamed_kv_bytes": 3458203648,
            "checkpoint_bytes": 944308224,
            "pages": 65536,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=1, context=32768, ngram_in_ram=True, host_memory=68719476736, ram_max=51539607552),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1600663296,
            "reserve_per_sequence": 317493248,
            "reserve_per_card": 317493248,
            "free_per_card": 19294650368,
            "experts_cached": 18857,
            "experts_total": 24576,
            "arena_bytes": 11703361600,
            "streamed_kv_bytes": 463470592,
            "checkpoint_bytes": 944308224,
            "pages": 8192,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
    (
        PlanOptions(slots=1, context=32768, working_per_card=1073741824, checkpoints=0),
        {
            "cards": 2,
            "usable_vram": 24385683456,
            "layers": 48,
            "weights_per_card": 2341860608,
            "draft_per_card": 831015936,
            "working_per_card": 1073741824,
            "reserve_per_sequence": 317493248,
            "reserve_per_card": 317493248,
            "free_per_card": 19821571840,
            "experts_cached": 19372,
            "experts_total": 24576,
            "arena_bytes": 10649465600,
            "streamed_kv_bytes": 463470592,
            "checkpoint_bytes": 0,
            "pages": 8192,
            "kv_resident_cells": 32768,
            "ngram_bytes": 28800138240,
        },
    ),
]
GOLDEN_ROWS = [
    ("sequences x context", "1 x 262,144"),
    ("cards", "2"),
    ("usable VRAM per card", "24,385,683,456 B (22.71 GiB)"),
    ("weights per card", "2,341,860,608 B (2.18 GiB)"),
    ("draft layer per card", "831,015,936 B (0.77 GiB)"),
    ("working memory per card (estimate)", "1,600,663,296 B (1.49 GiB)"),
    ("reserve per sequence per card", "504,664,064 B (0.47 GiB)"),
    ("free for experts per card", "19,107,479,552 B (17.80 GiB)"),
    ("experts cached", "18,674 of 24,576"),
    ("host arena", "12,077,852,800 B (11.25 GiB)"),
    ("streamed KV", "3,458,203,648 B (3.22 GiB)"),
    ("checkpoints", "8 x 118,038,528 B"),
    ("pinned host memory", "16,480,364,672 B (15.35 GiB)"),
    ("n-gram table", "28,800,138,240 B (26.82 GiB), read from the file"),
    ("host memory with working set (estimate)", "19,701,590,144 B (18.35 GiB)"),
    ("KV pages", "65,536 of 4 cells"),
]

GOLDEN_FIT = {
    "fit_card0": "Arc_Pro_B60,22.71GiB,IP_20.1.0,20_Xe_cores,PCIe_4.0_x8/22.71GiB/-",
    "fit_card1": "Arc_Pro_B60,22.71GiB,IP_20.1.0,20_Xe_cores,PCIe_4.0_x8/22.71GiB/-",
    "fit_experts": "-/389/389",
    "fit_working": "-/1.49GiB/1.49GiB",
    "fit_mem_available": "64.00GiB/18.35GiB/1.00GiB",
    "fit_memory_max": "-/-/-",
    "fit_ram_max": "-/-/-",
    "fit_arena": "-/11.25GiB/13.00GiB",
    "fit_pinned": "-/15.35GiB/-",
    "fit_swap": "8.00GiB,cgroup:-/-/-",
    "fit_memlock": "unlimited/-/-",
    "fit_cpu": "Test_CPU,32_CPUs,16_cores,quota:none/16_workers_by_cores/16_workers",
    "fit_host_read": "-/-/32.7GB/s",
    "fit_loader": "6.50GB/s/-/8_threads",
    "fit_ngram": "80us_a_4KiB_read/file/file",
    "fit_copy0": "-/-/19.0GB/s",
    "fit_copy1": "-/-/19.0GB/s",
}


@pytest.mark.parametrize(("options", "expected"), GOLDEN)
def test_qwen4exps_plan_is_what_it_was(options: PlanOptions, expected: dict[str, int]):
    p = plan(model_memory(TABLE, options.cards), options)
    assert {name: getattr(p, name) for name in expected} == expected
    assert p.tiers is None
    assert p.card_layer_experts == ()
    assert p.host_caches_bytes == 0


def test_qwen4exps_plan_rows_are_what_they_were():
    assert plan(model_memory(TABLE), PlanOptions()).rows() == GOLDEN_ROWS


def test_qwen4exps_fit_report_is_what_it_was():
    p = plan(model_memory(TABLE), PlanOptions(host_memory=64 * GIB))
    the_host = Host(
        mem_total=90 * GIB,
        mem_available=64 * GIB,
        swap_total=8 * GIB,
        cgroup_memory_max=None,
        cgroup_swap_max=None,
        memlock=None,
        cpus=32,
        cpu_quota=None,
        physical_cores=16,
        cpu_flags=frozenset(),
        cpu_model="Test CPU",
    )
    card = Card("Arc Pro B60", "0000:03:00.0", 24_385_683_456, 0xE20B, (20 << 22) | (1 << 14), 20, 4, 8, 16 * 10**9)
    facts = EngineFacts(
        cards=2,
        slots=1,
        max_context=262_144,
        pages=65_536,
        checkpoints=8,
        experts_cached=389,
        vram_used=(),
        vram_free=(),
        host_pinned=13 * GIB,
        timing_fd=-1,
        version="t",
        host_read_bytes_per_s=32.7e9,
        loader_threads=8,
        loader_bytes_per_s=6.5e9,
        ngram_read_us=80.0,
        copy_bytes_per_s=(19e9, 19e9),
        expert_budget=(p.free_per_card, p.free_per_card),
    )
    rows = (
        fit.card_rows([card, card], p, 389, p.working_per_card)
        + fit.host_rows(the_host, p, 13 * GIB, GIB)
        + fit.cpu_rows(the_host, 16, 32.7e9)
        + fit.storage_rows(p, facts)
        + fit.tier_rows(p, 389, GIB)
    )
    assert fit.info(rows) == GOLDEN_FIT


def three_tiers(**more: object) -> ModelMemory:
    """Four layers of 100 experts, 8 MiB each in the first two layers and 12 MiB in the last two; 3 GiB of weights and
    working memory leave each 4 GiB card 1 GiB for experts. The host's other figures are 1 GiB of rings and caches."""
    sizes = {
        "cards": 2,
        "weights_per_card": 2 * GIB,
        "draft_per_card": 0,
        "working_per_card": GIB,
        "layers": 4,
        "experts": 400,
        "expert_bytes": 10 * MIB,
        "kv_cell_per_card": 0,
        "indexer_cell_per_card": 0,
        "state_per_card": 0,
        "draft_kv_per_card": 0,
        "resident_cells": 0,
        "kv_cell_host": 0,
        "draft_kv_host": 0,
        "checkpoint_bytes": 0,
        "max_context": 4096,
        "ngram_bytes": 0,
        "reference_vram": 4 * GIB,
        "layer_expert_bytes": (8 * MIB, 8 * MIB, 12 * MIB, 12 * MIB),
        "ssd_tier": True,
        "engram_cache": 256 * MIB,
        "ssd_staging_ring": 256 * MIB,
        "prompt_host_ring": 512 * MIB,
        "routes_per_row": 6,
    }
    return ModelMemory(**{**sizes, **more})  # pyright: ignore[reportArgumentType]


def options(host_gib: float | None) -> PlanOptions:
    limit = None if host_gib is None else round(host_gib * GIB)
    return PlanOptions(context=4096, checkpoints=0, host_working=GIB, host_memory=limit)


def test_three_tiers_from_what_the_host_has():
    """1 GiB beside the working set, the rings and the headroom: 256 MiB a layer, in whole experts."""
    assert SSD_TIER_HEADROOM == 24 * GIB
    p = plan(three_tiers(), options(24 + 1 + 1 + 1))
    t = p.tiers
    assert t is not None
    assert t.vram == (64, 64, 42, 43)
    assert t.arena == (32, 32, 21, 21)
    assert t.ssd == (4, 4, 37, 36)
    assert p.card_layer_experts == (t.vram, t.vram)
    assert p.experts_cached == sum(t.vram)
    assert p.arena_bytes == t.bytes_of(t.arena) == 1016 * MIB
    assert t.bytes_of(t.ssd) == 940 * MIB
    assert p.host_bytes == p.arena_bytes + GIB + GIB
    assert any(name == "SSD tier, experts a layer" and value.startswith("4-37,") for name, value in p.rows())


def test_the_arena_is_capped_by_what_is_left_to_hold():
    p = plan(three_tiers(), options(100))
    t = p.tiers
    assert t is not None
    assert t.ssd == (0, 0, 0, 0)
    assert t.arena == (36, 36, 58, 57)
    assert p.arena_bytes == 1956 * MIB


def test_the_streamed_kv_and_checkpoints_come_out_of_the_arena():
    """512 MiB of KV (1,024 pages of 4 cells, 128 KiB a cell) and 512 MiB of checkpoints, on top of the host above."""
    m = three_tiers(kv_cell_host=128 * 1024, checkpoint_bytes=256 * MIB)
    o = PlanOptions(context=4096, checkpoints=2, host_working=GIB, host_memory=(24 + 1 + 1 + 1 + 1) * GIB)
    p = plan(m, o)
    assert p.streamed_kv_bytes + p.checkpoint_bytes == GIB
    assert p.arena_bytes == 1016 * MIB


@pytest.mark.parametrize("host_gib", [24 + 2, 3.0, 2.0, None])
def test_an_arena_that_does_not_fit_is_a_plan_with_all_on_the_ssd(host_gib: float | None):
    p = plan(three_tiers(), options(host_gib))
    t = p.tiers
    assert t is not None
    assert p.arena_bytes == 0
    assert t.arena == (0, 0, 0, 0)
    assert t.ssd == (36, 36, 58, 57)


def test_a_host_short_of_the_memory_outside_the_arena_is_still_refused():
    with pytest.raises(PlanError, match=r"Engram cache and staging rings 1\.0"):
        plan(three_tiers(), options(1.5))


def test_the_ram_max_limit_sizes_the_arena():
    o = PlanOptions(context=4096, checkpoints=0, host_working=GIB, host_memory=100 * GIB, ram_max=27 * GIB)
    assert plan(three_tiers(), o).arena_bytes == 1016 * MIB


def test_the_ngram_table_in_ram_comes_out_of_the_arena():
    o = PlanOptions(context=4096, checkpoints=0, host_working=GIB, host_memory=27 * GIB, ngram_in_ram=True)
    assert plan(three_tiers(ngram_bytes=GIB), o).arena_bytes == 0


def test_the_headroom_applies_only_with_an_ssd_tier():
    """Without an SSD tier the arena holds every expert VRAM does not, on a host under the headroom too."""
    p = plan(three_tiers(ssd_tier=False, engram_cache=0, ssd_staging_ring=0, prompt_host_ring=0), options(20))
    assert p.tiers is None
    assert p.card_layer_experts[0] == (64, 64, 42, 43)
    assert p.arena_bytes == 1956 * MIB


@pytest.mark.parametrize("fuller", [0, 1])
def test_unequal_cards_get_exact_weights(fuller: int):
    """One card has 512 MiB more for experts, in either position; a layer caches what the fuller card can hold."""
    weights = (2 * GIB, GIB + 512 * MIB) if fuller == 0 else (GIB + 512 * MIB, 2 * GIB)
    free = (GIB, GIB + 512 * MIB) if fuller == 0 else (GIB + 512 * MIB, GIB)
    held = ((64, 64, 42, 43), (96, 96, 64, 64)) if fuller == 0 else ((96, 96, 64, 64), (64, 64, 42, 43))
    p = plan(three_tiers(card_weights=weights), options(27))
    t = p.tiers
    assert t is not None
    assert p.card_weights == weights
    assert p.weights_per_card == 2 * GIB
    assert p.card_free == free
    assert p.free_per_card == GIB
    assert p.card_layer_experts == held
    assert t.vram == (64, 64, 42, 43)
    assert (f"weights on card {1 - fuller}", f"{GIB + 512 * MIB:,} B (1.50 GiB)") in p.rows()


@pytest.mark.parametrize("free", [(28 * MIB, 27 * MIB), (27 * MIB, 28 * MIB), (27 * MIB, 27 * MIB)])
def test_a_roomier_card_never_costs_vram_experts(free: tuple[int, int]):
    """The fill runs on the smallest budget: 1 MiB more on one card does not lose an expert that both cards can hold."""
    sizes = {"layers": 2, "experts": 200, "layer_expert_bytes": (14 * MIB, 16 * MIB)}
    p = plan(three_tiers(card_weights=(3 * GIB - free[0], 3 * GIB - free[1]), **sizes), options(None))
    assert p.card_free == free
    t = p.tiers
    assert t is not None
    assert t.vram == (1, 2)
    assert t.bytes_of(t.vram) // 2 <= min(free)


@pytest.mark.parametrize("weights", [(2 * GIB, GIB), (GIB, 2 * GIB)])
def test_unequal_cards_with_one_count(weights: tuple[int, int]):
    """A model without per-layer sizes keeps the one count, from the fuller card."""
    m = three_tiers(card_weights=weights, layer_expert_bytes=(), ssd_tier=False, engram_cache=0, ssd_staging_ring=0)
    p = plan(m, options(None))
    assert p.card_layer_experts == ()
    assert p.experts_cached == GIB // (5 * MIB)
    assert p.arena_bytes == (400 - p.experts_cached) * 10 * MIB


@pytest.mark.parametrize("weights", [(3 * GIB, GIB), (GIB, 3 * GIB)])
def test_a_card_short_names_the_fullest_cards_weights_and_prompt_staging(weights: tuple[int, int]):
    m = three_tiers(card_weights=weights, prompt_staging_per_card=GIB)
    with pytest.raises(PlanError, match=r"the weights take 3.00.*prompt staging 1.00"):
        plan(m, options(27))


@pytest.mark.parametrize(
    "more", [{"card_weights": (GIB,)}, {"layer_expert_bytes": (MIB,)}], ids=["card_weights", "layer_expert_bytes"]
)
def test_figures_of_the_wrong_length_are_an_error(more: dict[str, object]):
    with pytest.raises(ValueError, match=" for "):
        three_tiers(**more)


@pytest.mark.parametrize("experts", [2, 0])
def test_a_layer_with_no_expert_is_an_error(experts: int):
    with pytest.raises(ValueError, match="at least one"):
        three_tiers(experts=experts)


def test_the_report_at_uniform_routing():
    p = plan(three_tiers(), options(27))
    rows = {r.key: r for r in fit.tier_rows(p)}
    assert rows["tier_vram"].plan == "42-64_a_layer,2.00GiB"
    assert rows["tier_arena"].plan == "21-32_a_layer,0.99GiB"
    assert rows["tier_ssd"].plan == "4-37_a_layer,0.92GiB"
    # 6 x (0.04 + 0.04 + 0.37 + 0.36) reads, of 8 MiB in the first two layers and 12 MiB in the others.
    assert rows["ssd_token"].plan == f"4.9_reads,{fit.gib(round(56.4 * MIB))},uniform_routing"
    assert rows["process_huge"].host == ""
    assert rows["tier_vram"].core == ""


def test_the_report_from_a_hit_curve():
    """Half the experts serve 75% of routings: each tier's share follows the curve, and SSD reads shrink."""
    p = plan(three_tiers(hit_curve=((0.5, 0.75),)), options(27))
    t = p.tiers
    assert t is not None
    shares = t.routing_shares()
    assert shares is not None
    vram_at = [0.75 + (n / 100 - 0.5) * 0.5 if n >= 50 else n / 100 * 1.5 for n in t.vram]
    held_at = [0.75 + ((v + a) / 100 - 0.5) * 0.5 for v, a in zip(t.vram, t.arena, strict=True)]
    assert shares[0] == pytest.approx(sum(vram_at) / 4)
    assert shares[1] == pytest.approx((sum(held_at) - sum(vram_at)) / 4)
    assert shares[2] == pytest.approx(1 - sum(held_at) / 4)
    assert sum(shares) == pytest.approx(1)
    rows = {r.key: r for r in fit.tier_rows(p)}
    assert rows["tier_ssd"].plan.endswith(f",{shares[2]:.1%}_of_routings")
    assert "uniform" not in rows["ssd_token"].plan
    reads, _ = t.ssd_per_token()
    assert reads == pytest.approx(6 * 4 * shares[2])


def test_no_routes_given_leaves_the_reads_unknown():
    rows = {r.key: r for r in fit.tier_rows(plan(three_tiers(routes_per_row=0), options(27)))}
    assert rows["ssd_token"].plan == ""


def test_the_cores_count_is_the_truth_in_the_report():
    """The core caches 40 a layer: VRAM scales to it, the arena keeps its experts, the SSD takes the rest."""
    p = plan(three_tiers(), options(27))
    t = p.tiers
    assert t is not None
    core = t.with_core(40)
    assert core.vram == (48, 48, 32, 32)
    assert core.arena == t.arena
    assert core.ssd == (20, 20, 47, 47)
    row = next(r for r in fit.tier_rows(p, 40) if r.key == "tier_vram")
    assert row.plan.startswith("42-64_a_layer")
    assert row.core.startswith("32-48_a_layer")
    assert t.with_core(1000).vram == (100,) * 4
    assert t.with_core(1000).arena == (0,) * 4


def test_huge_pages_are_reported_as_the_whole_process():
    """More huge pages than arena is no coverage figure: an n-gram table in RAM counts too."""
    p = plan(three_tiers(), options(27))
    row = next(r for r in fit.tier_rows(p, 50, 2 * GIB) if r.key == "process_huge")
    assert row.name == "process memory in huge pages"
    assert row.host == "2.00GiB"


def test_the_report_after_a_load_holds_the_tiers(monkeypatch: pytest.MonkeyPatch):
    """The core caches 40 a layer, and the process has 3 GiB in huge pages."""
    monkeypatch.setattr(fit, "anon_huge_pages", lambda: 3 * GIB)
    p = plan(three_tiers(), options(27))
    the_host = Host(
        mem_total=90 * GIB,
        mem_available=64 * GIB,
        swap_total=0,
        cgroup_memory_max=None,
        cgroup_swap_max=None,
        memlock=None,
        cpus=4,
        cpu_quota=None,
        physical_cores=2,
        cpu_flags=frozenset(),
        cpu_model="Test CPU",
    )
    card = Card("Arc Pro B60", "0000:03:00.0", 4 * GIB, 0xE20B, (20 << 22) | (1 << 14), 20, 4, 8, 16 * 10**9)
    facts = EngineFacts(
        cards=2,
        slots=1,
        max_context=4096,
        pages=1024,
        checkpoints=0,
        experts_cached=40,
        vram_used=(),
        vram_free=(),
        host_pinned=GIB,
        timing_fd=-1,
        version="t",
    )
    rows = {r.key: r for r in report(the_host, [card, card], p, 2, facts)[0]}
    assert rows["tier_vram"].core.startswith("32-48_a_layer")
    assert rows["process_huge"].host == "3.00GiB"


def test_prompt_staging_is_priced_in_experts():
    """10 MiB mean expert, 5 MiB a card: 400 MiB a card over 4 layers is 20; 512 MiB of ring over 4 layers is 12."""
    m = three_tiers(prompt_staging_per_card=400 * MIB)
    rows = {r.key: r for r in fit.tier_rows(plan(m, options(27)))}
    assert rows["prompt_staging"].plan == "20_VRAM,12_arena"
    assert plan(m, options(27)).free_per_card == GIB - 400 * MIB


def test_anon_huge_pages(tmp_path: Path):
    rollup = tmp_path / "smaps_rollup"
    rollup.write_text("Rss:   1024 kB\nAnonHugePages:   2048 kB\n")
    assert fit.anon_huge_pages(rollup) == 2 * MIB
    assert fit.anon_huge_pages(tmp_path / "none") is None


def test_every_new_figure_is_unset_for_flash_next():
    m = model_memory(TABLE)
    assert (m.card_weights, m.layer_expert_bytes, m.hit_curve, m.ssd_tier) == ((), (), (), False)
    assert (m.engram_cache, m.ssd_staging_ring, m.prompt_staging_per_card, m.prompt_host_ring, m.routes_per_row) == (
        0,
        0,
        0,
        0,
        0,
    )
