"""What the host gives the engine, read once before the memory plan (ADR-0057). It reads and never writes.

Each value comes from the kernel's own files: `/proc/meminfo`, the process's cgroup (v2) under `/sys/fs/cgroup`,
`/proc/cpuinfo`, the CPU topology in `/sys/devices/system/cpu`, the PCI devices in `/sys/bus/pci`, `sched_getaffinity`
and `getrlimit`. A file that cannot be read gives None, and the plan then takes no limit from it. Inside a container
only the container's own cgroup is visible, so a limit set on a parent cgroup is not read.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import resource
from dataclasses import dataclass
from pathlib import Path

GIB = 2**30


@dataclass(frozen=True, slots=True)
class Host:
    mem_total: int | None
    """`MemTotal`, bytes."""
    mem_available: int | None
    """`MemAvailable`: what the host can give without swapping, bytes."""
    swap_total: int | None
    cgroup_memory_max: int | None
    """The cgroup's `memory.max`, bytes; None for "max" or no cgroup."""
    cgroup_swap_max: int | None
    """The cgroup's `memory.swap.max`, bytes; None for "max" or no cgroup."""
    memlock: int | None
    """`RLIMIT_MEMLOCK`'s soft limit, bytes; None for unlimited."""
    cpus: int
    """CPUs the process may run on (`sched_getaffinity`)."""
    cpu_quota: float | None
    """The cgroup's `cpu.max` as CPUs (quota / period); None for "max"."""
    physical_cores: int
    """Cores among the CPUs the process may run on, each counted once whatever its threads."""
    cpu_flags: frozenset[str]
    """The first CPU's flags in `/proc/cpuinfo`."""
    cpu_model: str
    huge_pages_free: int | None = None
    """The reserved huge pages (`vm.nr_hugepages`) no process holds or has reserved: (`HugePages_Free` -
    `HugePages_Rsvd`) x `Hugepagesize`, bytes."""
    huge_page_size: int | None = None
    """`Hugepagesize`: the default size of a reserved huge page, bytes."""

    @property
    def memory_limit(self) -> int | None:
        """The most host memory the engine can take: the smaller of `MemAvailable` and `memory.max`."""
        known = [v for v in (self.mem_available, self.cgroup_memory_max) if v is not None]
        return min(known) if known else None

    def cpu_workers(self) -> tuple[int, str]:
        """The CPU's expert workers, the engine thread among them: one per core the process can keep busy, so none
        waits for its quota or shares a core (`ENGINE_CPU_WORKERS` overrides it); with what bounds the count."""
        n = min(self.usable_cpus, self.physical_cores)
        if n == self.physical_cores:
            return n, "cores"
        return n, "cpu.max" if self.cpu_quota is not None and n == self.usable_cpus < self.cpus else "affinity"

    @property
    def usable_cpus(self) -> int:
        """CPUs the process can keep busy at once: its affinity, cut to the cgroup's quota."""
        if self.cpu_quota is None:
            return self.cpus
        return max(1, min(self.cpus, math.ceil(self.cpu_quota)))


@dataclass(frozen=True, slots=True)
class Card:
    """One card the engine can use, from the core's `engine_cards` (Level Zero), with its link from `slot_link`."""

    name: str
    pci: str
    memory: int
    """Device-local memory, bytes."""
    device_id: int
    ip_version: int
    """Level Zero's packed IP version; 0 when not reported."""
    xe_cores: int
    pcie_gen: int
    """The PCIe generation of the slowest link between the card and the CPU (`slot_link`); -1: unknown."""
    pcie_width: int
    """Lanes of the narrowest link between the card and the CPU; -1: unknown."""
    pcie_bytes_per_s: int
    """What `pcie_gen` and `pcie_width` carry after line encoding, bytes/s; -1: unknown."""

    @property
    def ip(self) -> str:
        """The IP version as architecture.release.revision, the form `ocloc` takes: 20.1.0 for a B60."""
        v = self.ip_version
        return f"{v >> 22}.{(v >> 14) & 0xFF}.{v & 0x3F}" if v else "unknown"

    @property
    def link(self) -> str:
        return f"PCIe {self.pcie_gen}.0 x{self.pcie_width}" if self.pcie_gen > 0 and self.pcie_width > 0 else "unknown"


def _read(path: Path) -> str | None:
    try:
        return path.read_text()
    except OSError:
        return None


def meminfo(path: Path) -> dict[str, int]:
    """`/proc/meminfo`'s values in bytes."""
    out: dict[str, int] = {}
    for line in (_read(path) or "").splitlines():
        key, _, rest = line.partition(":")
        fields = rest.split()
        if fields and fields[0].isdigit():
            out[key] = int(fields[0]) * (1024 if len(fields) > 1 and fields[1] == "kB" else 1)
    return out


def huge_pages_free(mi: dict[str, int]) -> int | None:
    """The free reserved huge pages in `/proc/meminfo`'s values (`meminfo`), bytes."""
    free, reserved, size = mi.get("HugePages_Free"), mi.get("HugePages_Rsvd", 0), mi.get("Hugepagesize")
    if free is None or size is None:
        return None
    return max(0, free - reserved) * size


def cgroup_dir(proc_cgroup: Path, root: Path) -> Path | None:
    """The process's cgroup v2 directory: the `0::` line of `/proc/self/cgroup` under the mount."""
    for line in (_read(proc_cgroup) or "").splitlines():
        if line.startswith("0::"):
            d = root / line[3:].lstrip("/")
            return d if d.is_dir() else None
    return None


def _limit(path: Path) -> int | None:
    text = _read(path)
    if text is None or text.strip() in ("", "max"):
        return None
    try:
        return int(text.split()[0])
    except ValueError:
        return None


def cpu_quota(path: Path) -> float | None:
    """`cpu.max` ("quota period", or "max period") as CPUs."""
    fields = (_read(path) or "").split()
    if len(fields) != 2 or fields[0] == "max":
        return None
    try:
        return int(fields[0]) / int(fields[1])
    except ValueError, ZeroDivisionError:
        return None


def physical_cores(cpus: set[int], sys_cpu: Path) -> int:
    """Distinct (package, core) pairs among cpus; their count when the topology cannot be read."""
    cores: set[tuple[str, str]] = set()
    for c in cpus:
        topo = sys_cpu / f"cpu{c}" / "topology"
        package, core = _read(topo / "physical_package_id"), _read(topo / "core_id")
        if package is None or core is None:
            return len(cpus)
        cores.add((package.strip(), core.strip()))
    return len(cores) or len(cpus)


def cpuinfo(path: Path) -> tuple[frozenset[str], str]:
    """The first CPU's flags and model name."""
    flags: frozenset[str] = frozenset()
    model = ""
    for line in (_read(path) or "").splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "flags" and not flags:
            flags = frozenset(value.split())
        elif key == "model name" and not model:
            model = value.strip()
        if flags and model:
            break
    return flags, model


_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
_GEN_BY_GT_S = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}
_LANE_BYTES_PER_S = {
    1: 250e6,
    2: 500e6,
    3: 8e9 / 8 * 128 / 130,
    4: 16e9 / 8 * 128 / 130,
    5: 32e9 / 8 * 128 / 130,
    6: 64e9 / 8 * 242 / 256,
}
"""One lane's rate after line encoding: 8b/10b to PCIe 2.0, 128b/130b to 5.0, flits from 6.0."""


def _link_gen(text: str | None) -> int | None:
    """sysfs's link speed ("16.0 GT/s PCIe") as a PCIe generation."""
    try:
        return _GEN_BY_GT_S.get(float((text or "").split()[0]))
    except ValueError, IndexError:
        return None


def _link_width(text: str | None) -> int | None:
    try:
        width = int((text or "").strip())
    except ValueError:
        return None
    return width if width > 0 else None


def slot_link(card: Card, root: Path = Path("/")) -> Card:
    """The card with its PCIe link taken as the slowest on the path to the CPU's root port, from sysfs. Level Zero
    gives the card's own link, which on a card with a PCIe switch on board is the link to that switch. A field sysfs
    does not give keeps Level Zero's value. root: where `/sys` is, for tests."""
    try:
        dev = (root / "sys/bus/pci/devices" / card.pci).resolve(strict=True)
    except OSError, RuntimeError:
        return card
    gens: list[int] = []
    widths: list[int] = []
    # Each port's link files describe one link on the path: an endpoint's or a switch's upstream port the link above
    # it, a root port's or a switch's downstream port the link below it. So the path's slowest is the least over them.
    while _BDF.match(dev.name):
        # Speed from the most both ends support: a link idles at a lower speed to save power, so the trained speed read
        # at load can be below what copies get. Width is the trained one; links keep their width when idle.
        if (gen := _link_gen(_read(dev / "max_link_speed"))) is not None:
            gens.append(gen)
        if (width := _link_width(_read(dev / "current_link_width"))) is not None:
            widths.append(width)
        dev = dev.parent
    if not gens and not widths:
        return card
    gen = min(gens) if gens else card.pcie_gen
    width = min(widths) if widths else card.pcie_width
    rate = round(_LANE_BYTES_PER_S[gen] * width) if gen in _LANE_BYTES_PER_S and width > 0 else -1
    return dataclasses.replace(card, pcie_gen=gen, pcie_width=width, pcie_bytes_per_s=rate)


def read(root: Path = Path("/"), affinity: set[int] | None = None) -> Host:
    """The host as this process sees it. root: where `/proc` and `/sys` are; affinity: the CPUs the process may use,
    by default its own. Both are for tests."""
    mi = meminfo(root / "proc/meminfo")
    cg = cgroup_dir(root / "proc/self/cgroup", root / "sys/fs/cgroup")
    affinity = os.sched_getaffinity(0) if affinity is None else affinity
    soft, _ = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    flags, model = cpuinfo(root / "proc/cpuinfo")
    return Host(
        mem_total=mi.get("MemTotal"),
        mem_available=mi.get("MemAvailable"),
        swap_total=mi.get("SwapTotal"),
        huge_pages_free=huge_pages_free(mi),
        huge_page_size=mi.get("Hugepagesize"),
        cgroup_memory_max=_limit(cg / "memory.max") if cg else None,
        cgroup_swap_max=_limit(cg / "memory.swap.max") if cg else None,
        memlock=None if soft == resource.RLIM_INFINITY else soft,
        cpus=len(affinity),
        cpu_quota=cpu_quota(cg / "cpu.max") if cg else None,
        physical_cores=physical_cores(affinity, root / "sys/devices/system/cpu"),
        cpu_flags=flags,
        cpu_model=model,
    )
