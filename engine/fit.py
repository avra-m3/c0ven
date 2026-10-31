"""The fit report: what the host and the cards give, beside the plan's figure that depends on each and the core's figure
after the load (ADR-0057). It is printed at each load and sent in INFO, so `/metrics` carries it as `fit_*` facts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from engine.api.interface import EngineFacts
from engine.host import Card, Host
from engine.memory_plan import GIB, MemoryPlan

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


EXPERTS_TOLERANCE = 0.05
"""A core's expert count further than this share from the plan's is noted: the plan's card figures are estimates."""


def card_rows(cards: list[Card], plan: MemoryPlan, experts_core: int | None = None) -> list[Row]:
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
    return rows


def card_notes(plan: MemoryPlan, experts_core: int) -> list[str]:
    """What the core's figures say about the plan's, after the load."""
    p = plan.experts_per_layer
    if abs(experts_core - p) > max(8, EXPERTS_TOLERANCE * p):
        return [
            f"the core caches {experts_core} experts a layer and the plan {p}: the plan's figures for the cards, "
            "and the host memory planned from them, are off by that much"
        ]
    return []


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
            "RAM" if plan.options.ngram_in_ram else "file",
            "RAM" if facts.ngram_in_ram else "file",
        ),
    ]
    rows += [
        Row(f"copy{i}", f"card {i} copy from host memory", "", "", f"{r / 1e9:.1f}GB/s")
        for i, r in enumerate(facts.copy_bytes_per_s)
    ]
    return rows


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
