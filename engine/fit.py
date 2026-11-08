"""The fit report: what the host and the cards give, beside the plan's figure that depends on each and the core's figure
after the load (ADR-0057). It is printed at each load and sent in INFO, so `/metrics` carries it as `fit_*` facts. A
model with an SSD tier adds a row for each tier.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from engine.api.interface import EngineFacts
from engine.host import Card, Host
from engine.memory_plan import EXPERTS_TOLERANCE, EXPERTS_TOLERANCE_MIN, GIB, MemoryPlan, Tiers

UNKNOWN = "-"


@dataclass(frozen=True, slots=True)
class Row:
    key: str
    """The INFO key's suffix: `fit_<key>`."""
    name: str
    host: str
    """What was read from the host or the cards."""
    plan: str = ""
    """The plan's figure that depends on it."""
    core: str = ""
    """The core's figure after the load."""


def gib(n: int | None) -> str:
    return UNKNOWN if n is None else f"{n / GIB:.2f}GiB"


def rss(status: Path = Path("/proc/self/status")) -> int | None:
    """The process's resident memory, bytes."""
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError, ValueError, IndexError:
        return None
    return None


def anon_huge_pages(rollup: Path = Path("/proc/self/smaps_rollup")) -> int | None:
    """The process's anonymous memory in huge pages, bytes."""
    try:
        for line in rollup.read_text().splitlines():
            if line.startswith("AnonHugePages:"):
                return int(line.split()[1]) * 1024
    except OSError, ValueError, IndexError:
        return None
    return None


def host_rows(host: Host, plan: MemoryPlan, arena_core: int | None = None, rss_core: int | None = None) -> list[Row]:
    o = plan.options
    rows = [
        Row("mem_available", "host memory available", gib(host.mem_available), gib(plan.host_bytes), gib(rss_core)),
        Row("memory_max", "cgroup memory.max", gib(host.cgroup_memory_max)),
        Row("ram_max", "--ram-max", gib(o.ram_max)),
        Row("arena", "arena", "", gib(plan.arena_bytes), gib(arena_core)),
        Row("pinned", "pinned host memory", "", gib(plan.pinned_bytes)),
        Row("swap", "swap", f"{gib(host.swap_total)},cgroup:{gib(host.cgroup_swap_max)}"),
        Row("memlock", "RLIMIT_MEMLOCK", gib(host.memlock) if host.memlock is not None else "unlimited"),
    ]
    return rows


WORKING_TOLERANCE = 256 * 2**20
"""A core's working memory further than this from the plan's is noted."""


def card_rows(
    cards: list[Card], plan: MemoryPlan, experts_core: int | None = None, working_core: int | None = None
) -> list[Row]:
    rows = [
        Row(
            f"card{i}",
            f"card {i}",
            f"{c.name.replace(' ', '_')},{gib(c.memory)},IP_{c.ip},{c.xe_cores}_Xe_cores,{c.link.replace(' ', '_')}",
            gib(plan.usable_vram) if i < plan.cards else "unused",
        )
        for i, c in enumerate(cards)
    ]
    rows.append(
        Row(
            "experts",
            "experts per layer",
            "",
            str(plan.experts_per_layer),
            str(experts_core) if experts_core is not None else "",
        )
    )
    rows.append(
        Row(
            "working",
            "working memory per card",
            "",
            gib(plan.working_per_card),
            gib(working_core) if working_core is not None else "",
        )
    )
    return rows


def card_notes(plan: MemoryPlan, experts_core: int, working_core: int | None = None) -> list[str]:
    """What the core's figures say about the plan's, after the load."""
    out: list[str] = []
    p = plan.experts_per_layer
    if abs(experts_core - p) > max(EXPERTS_TOLERANCE_MIN, EXPERTS_TOLERANCE * p):
        out.append(
            f"the core caches {experts_core} experts a layer and the plan {p}: the plan's figures for the cards, "
            "and the host memory planned from them, are off by that much"
        )
    if working_core is not None and abs(working_core - plan.working_per_card) > WORKING_TOLERANCE:
        out.append(
            f"the core's working memory per card is {gib(working_core)} and the plan's {gib(plan.working_per_card)}: "
            "with ENGINE_CACHE_DIR set, the next start with these options plans with the core's"
        )
    return out


def cpu_rows(host: Host, workers: int, read_bytes_per_s: float) -> list[Row]:
    n, source = host.cpu_workers()
    quota = f"{host.cpu_quota:g}" if host.cpu_quota is not None else "none"
    return [
        Row(
            "cpu",
            "CPU",
            f"{host.cpu_model.replace(' ', '_')},{host.cpus}_CPUs,{host.physical_cores}_cores,quota:{quota}",
            f"{n}_workers_by_{source}",
            f"{workers}_workers",
        ),
        Row(
            "host_read",
            "the workers' read of the arena",
            "",
            core=f"{read_bytes_per_s / 1e9:.1f}GB/s" if read_bytes_per_s else "",
        ),
    ]


def _tier_text(t: Tiers, experts: tuple[int, ...], share: float | None) -> str:
    low, high = min(experts), max(experts)
    count = str(low) if low == high else f"{low}-{high}"
    return f"{count}_a_layer,{gib(t.bytes_of(experts))}" + (f",{share:.1%}_of_routings" if share is not None else "")


def _tiers_text(t: Tiers) -> list[str]:
    """The VRAM, arena and SSD figures, then the SSD reads a token: each as one report cell."""
    shares = t.routing_shares() or (None, None, None)
    cells = [_tier_text(t, n, s) for n, s in zip((t.vram, t.arena, t.ssd), shares, strict=True)]
    if not t.routes_per_row:
        return [*cells, ""]
    reads, size = t.ssd_per_token()
    basis = "" if t.hit_curve else ",uniform_routing"
    return [*cells, f"{reads:.1f}_reads,{gib(round(size))}{basis}"]


def tier_rows(
    plan: MemoryPlan,
    experts_core: int | None = None,
    huge_core: int | None = None,
) -> list[Row]:
    """A model with an SSD tier: each tier's experts a layer and bytes, the share of routings it expects when a hit
    curve is given, and the SSD reads and bytes a token at one row. The core's count, when given, is the truth: its
    column follows from it. huge_core: the whole process's anonymous memory in huge pages, bytes, which holds more than
    the arena, so it bounds the arena's coverage and does not measure it. Empty for any other model."""
    t = plan.tiers
    if t is None:
        return []
    planned = _tiers_text(t)
    core = _tiers_text(t.with_core(experts_core)) if experts_core is not None else [""] * len(planned)
    rows = [
        Row(key, name, "", p, c)
        for key, name, p, c in zip(
            ("tier_vram", "tier_arena", "tier_ssd", "ssd_token"),
            ("VRAM tier", "arena tier", "SSD tier", "SSD reads a token, one row"),
            planned,
            core,
            strict=True,
        )
    ]
    rows.append(Row("process_huge", "process memory in huge pages", gib(huge_core) if huge_core is not None else ""))
    if plan.prompt_staging_per_card or plan.prompt_host_ring_bytes:
        mean = t.bytes_of((1,) * len(t.vram)) // len(t.vram)
        in_vram = plan.prompt_staging_per_card // plan.layers // max(1, -(-mean // plan.cards))
        in_arena = plan.prompt_host_ring_bytes // plan.layers // max(1, mean)
        rows.append(
            Row("prompt_staging", "experts a layer prompt staging costs", "", f"{in_vram}_VRAM,{in_arena}_arena")
        )
    return rows


LOADER_REFERENCE_BYTES_PER_S = 6.5e9
"""The test machine's load read rate (docs/measurements/MEASUREMENTS.md#2026-10-16-load-read-rates): a disk under half
of it is noted."""


def storage_rows(plan: MemoryPlan, facts: EngineFacts) -> list[Row]:
    rows = [
        Row(
            "loader",
            "model file read",
            f"{facts.loader_bytes_per_s / 1e9:.2f}GB/s" if facts.loader_bytes_per_s else "",
            "",
            f"{facts.loader_threads}_threads" if facts.loader_threads else "",
        ),
        Row(
            "ngram",
            "n-gram table",
            f"{facts.ngram_read_us:.0f}us_a_4KiB_read" if facts.ngram_read_us else "",
            ngram_place(plan.options.ngram_in_ram, plan.ngram_cache_rows, plan.options.ngram_hugetlb),
            ngram_place(facts.ngram_in_ram, facts.ngram_cache_rows, facts.ngram_huge_bytes),
        ),
    ]
    if (pool := plan.options.hugetlb_pool) is not None:
        rows.append(
            Row(
                "hugetlb",
                "reserved huge pages",
                gib(pool),
                f"arena_{gib(plan.arena_hugetlb_claim)},ngram_{gib(plan.options.ngram_hugetlb)}",
                f"ngram_{gib(facts.ngram_huge_bytes)}",
            )
        )
    rows += [
        Row(f"copy{i}", f"card {i} copy from host memory", "", "", f"{r / 1e9:.1f}GB/s")
        for i, r in enumerate(facts.copy_bytes_per_s)
    ]
    return rows


def ngram_place(in_ram: bool, cache_rows: int, huge: int) -> str:
    """Where the n-gram table is, as one report cell: `RAM` or `file`; with bytes in reserved huge pages, then the
    cache's rows when the table is read from the file, and those bytes."""
    cells = ["RAM" if in_ram else "file"]
    if huge and cache_rows and not in_ram:
        cells.append(f"cache_{cache_rows / 1e6:.1f}M_rows")
    if huge:
        cells.append(f"{gib(huge)}_huge")
    return ",".join(cells)


def storage_notes(facts: EngineFacts) -> list[str]:
    if 0 < facts.loader_bytes_per_s < LOADER_REFERENCE_BYTES_PER_S / 2:
        return [
            f"the model file reads at {facts.loader_bytes_per_s / 1e9:.2f} GB/s, under half the test machine's: loads "
            "and wakes take longer"
        ]
    return []


def info(rows: list[Row]) -> dict[str, str]:
    """The rows as INFO facts: `fit_<key>=host/plan/core`, a part left out where there is none."""
    out: dict[str, str] = {}
    for r in rows:
        parts = [r.host or UNKNOWN, r.plan or UNKNOWN, r.core or UNKNOWN]
        out[f"fit_{r.key}"] = "/".join(p.replace(" ", "_") for p in parts)
    return out


def write(rows: list[Row], notes: list[str], stream: TextIO = sys.stderr) -> None:
    """The table, one line a row, then the notes, for the load's log."""
    width = max(len(r.name) for r in rows)
    stream.write("engine: fit report, each row what was read / the plan's figure / the core's after the load:\n")
    for r in rows:
        stream.write(f"engine:   {r.name:<{width}}  {r.host or UNKNOWN} / {r.plan or UNKNOWN} / {r.core or UNKNOWN}\n")
    for note in notes:
        stream.write(f"engine: fit report: {note}\n")
    stream.flush()


def notes(host: Host, unlocked: bool) -> list[str]:
    """What the host's settings mean for the load, once a load."""
    out: list[str] = []
    swap = (host.swap_total or 0) > 0 and host.cgroup_swap_max != 0
    if swap and unlocked:
        out.append("the host has swap and the arena is not locked: a swapped-out expert is read back on each miss")
    return out
