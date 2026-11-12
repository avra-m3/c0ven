"""The n-gram table in the reserved huge pages the arena leaves (ADR-0070): the plan, `--ngram-ram`'s modes and the fit
report, on a host read from a made-up /proc/meminfo."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from engine import fit, host
from engine.api.interface import EngineFacts
from engine.memory_plan import (
    ARENA_PIECE,
    GIB,
    NGRAM_CACHE_ROWS,
    MemoryPlan,
    PlanOptions,
    ngram_auto,
    ngram_fits,
    plan,
)
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.tensors import from_tsv
from engine.process import ngram_placement
from tests.helpers import plan_options

MODEL = model_memory(from_tsv())
ROW = MODEL.ngram_row_bytes + 8
"""A cache row and its tag, bytes."""
MEM_TOTAL = 125 * GIB


def a_host(tmp_path: Path, free_pages: int = 36, page_kib: int = 1024 * 1024, available_gib: int = 14) -> host.Host:
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/meminfo").write_text(
        f"MemTotal: {MEM_TOTAL // 1024} kB\nMemAvailable: {available_gib * 1024 * 1024} kB\n"
        f"HugePages_Total: {free_pages}\nHugePages_Free: {free_pages}\nHugePages_Rsvd: 0\n"
        f"Hugepagesize: {page_kib} kB\n"
    )
    return host.read(tmp_path, affinity={0})


def options_for(h: host.Host, hugepages: bool = True, **more: object) -> PlanOptions:
    """Three slots of 262,144 positions on cards whose arena is about 17 GiB at the plan's count."""
    o = plan_options(
        slots=3,
        context=262_144,
        usable_vram=24 * GIB,
        host_memory=h.memory_limit,
        hugetlb_pool=(h.huge_pages_free or 0) if hugepages else None,
        huge_page=h.huge_page_size or 2 * 2**20,
    )
    return replace(o, **more)  # pyright: ignore[reportArgumentType]


def placed(o: PlanOptions, mode: str = "auto") -> MemoryPlan:
    in_ram, huge = ngram_placement(mode, plan(MODEL, o), MEM_TOTAL)
    return plan(MODEL, replace(o, ngram_in_ram=in_ram, ngram_hugetlb=huge))


def test_the_model_gives_the_tables_rows():
    assert MODEL.ngram_row_bytes == 90
    assert MODEL.ngram_bytes == 320_001_536 * 90


def test_the_host_gives_the_huge_page_size(tmp_path: Path):
    h = a_host(tmp_path)
    assert h.huge_pages_free == 36 * GIB
    assert h.huge_page_size == GIB


def test_the_arena_keeps_its_pieces_at_its_largest(tmp_path: Path):
    p = plan(MODEL, options_for(a_host(tmp_path)))
    assert 17 * GIB < p.arena_bytes < 18 * GIB
    # The bound: 16 experts a layer (5% of 320) fewer cached than the plan's count.
    assert p.experts_per_layer == 320
    assert p.arena_bound_bytes == (512 - 304) * 48 * MODEL.expert_bytes
    assert p.arena_hugetlb_claim == (-(-p.arena_bound_bytes // ARENA_PIECE) + 1) * ARENA_PIECE == 21 * GIB
    assert p.hugetlb_left == 15 * GIB


def test_the_cache_fills_the_pages_the_arena_leaves(tmp_path: Path):
    p = placed(options_for(a_host(tmp_path)))
    assert not p.options.ngram_in_ram
    assert p.options.ngram_hugetlb == 15 * GIB
    assert p.ngram_cache_rows == 15 * GIB // ROW > 160_000_000
    assert p.arena_hugetlb_claim + p.options.ngram_hugetlb <= 36 * GIB
    # The cache's pages are outside MemAvailable: the host total is as with the table in the file.
    assert p.host_bytes == plan(MODEL, options_for(a_host(tmp_path / "b"))).host_bytes


def test_the_whole_table_goes_in_huge_pages_when_they_hold_it(tmp_path: Path):
    p = placed(options_for(a_host(tmp_path, free_pages=60)))
    assert p.options.ngram_in_ram
    assert p.options.ngram_hugetlb == 27 * GIB
    assert p.ngram_host_bytes == 0
    assert p.host_bytes == plan(MODEL, options_for(a_host(tmp_path / "b", free_pages=60))).host_bytes


def test_the_table_is_split_when_the_host_holds_the_rest_under_the_margin(tmp_path: Path):
    h = a_host(tmp_path, available_gib=100)
    p = placed(options_for(h))
    assert p.options.ngram_in_ram
    assert p.options.ngram_hugetlb == 15 * GIB
    assert p.ngram_host_bytes == MODEL.ngram_bytes - 15 * GIB
    # The margin covers only the part outside the reserved pages: the whole table would not fit in ordinary memory.
    assert not ngram_fits(plan(MODEL, options_for(h)), MEM_TOTAL)


def test_two_mib_pages_round_to_whole_pages(tmp_path: Path):
    p = placed(options_for(a_host(tmp_path, free_pages=36 * 512 + 3, page_kib=2048)))
    # The arena's claim is whole pieces; the three pages past them go to the cache.
    assert p.options.ngram_hugetlb == 15 * GIB + 3 * 2 * 2**20


def test_too_few_pages_for_more_than_the_default_cache_take_none(tmp_path: Path):
    p = plan(MODEL, options_for(a_host(tmp_path, free_pages=21)))
    assert p.hugetlb_left == 0
    assert ngram_auto(p, MEM_TOTAL) == (False, 0)
    assert p.ngram_cache_rows == NGRAM_CACHE_ROWS


@pytest.mark.parametrize("available_gib", [40, 100, 120])
def test_without_hugepages_auto_is_as_before(tmp_path: Path, available_gib: int):
    p = plan(MODEL, options_for(a_host(tmp_path, available_gib=available_gib), hugepages=False))
    assert ngram_placement("auto", p, MEM_TOTAL) == (ngram_fits(p, MEM_TOTAL), 0)
    assert p.hugetlb_left == 0
    assert p.arena_hugetlb_claim == 0


@pytest.mark.parametrize("hugepages", [False, True])
def test_on_and_off_take_no_huge_pages(tmp_path: Path, hugepages: bool):
    p = plan(MODEL, options_for(a_host(tmp_path, available_gib=40), hugepages=hugepages))
    assert ngram_placement("on", p, MEM_TOTAL) == (True, 0)
    assert ngram_placement("off", p, MEM_TOTAL) == (False, 0)


def facts(**more: object) -> EngineFacts:
    base = EngineFacts(
        cards=2,
        slots=3,
        max_context=262_144,
        pages=196_608,
        checkpoints=8,
        experts_cached=320,
        vram_used=(),
        vram_free=(),
        host_pinned=17 * GIB,
        timing_fd=-1,
        version="t",
        ngram_read_us=76.0,
        ngram_cache_rows=NGRAM_CACHE_ROWS,
    )
    return replace(base, **more)  # pyright: ignore[reportArgumentType]


def test_the_fit_report_says_where_the_table_went(tmp_path: Path):
    p = placed(options_for(a_host(tmp_path)))
    rows = fit.info(fit.storage_rows(p, facts(ngram_cache_rows=p.ngram_cache_rows, ngram_huge_bytes=15 * GIB)))
    assert (
        rows["fit_ngram"]
        == "76us_a_4KiB_read/file,cache_164.3M_rows,15.00GiB_huge/file,cache_164.3M_rows,15.00GiB_huge"
    )
    assert rows["fit_hugetlb"] == "36.00GiB/arena_21.00GiB,ngram_15.00GiB/ngram_15.00GiB"
    whole = placed(options_for(a_host(tmp_path / "b", free_pages=60)))
    assert fit.info(fit.storage_rows(whole, facts(ngram_in_ram=True, ngram_huge_bytes=27 * GIB)))["fit_ngram"] == (
        "76us_a_4KiB_read/RAM,27.00GiB_huge/RAM,27.00GiB_huge"
    )


def test_the_fit_report_without_hugepages_is_as_before(tmp_path: Path):
    p = placed(options_for(a_host(tmp_path, available_gib=40), hugepages=False))
    rows = fit.info(fit.storage_rows(p, facts()))
    assert rows["fit_ngram"] == "76us_a_4KiB_read/file/file"
    assert "fit_hugetlb" not in rows


def test_the_plan_rows_name_both_claims(tmp_path: Path):
    p = placed(options_for(a_host(tmp_path)))
    rows = dict(p.rows())
    assert rows["n-gram cache"] == f"{p.ngram_cache_rows:,} rows"
    assert rows["reserved huge pages kept for the arena at its largest"].endswith("(21.00 GiB)")
    assert rows["n-gram table or cache in reserved huge pages"].endswith("(15.00 GiB)")
    without = plan(MODEL, options_for(a_host(tmp_path / "b", available_gib=40), hugepages=False))
    assert "n-gram cache" not in dict(without.rows())
