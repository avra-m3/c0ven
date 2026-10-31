"""What the host gives the engine, read once before the memory plan (ADR-0057). It reads and never writes.

Each value comes from the kernel's own files: `/proc/meminfo`, the process's cgroup (v2) under `/sys/fs/cgroup`,
`/proc/cpuinfo`, the CPU topology in `/sys/devices/system/cpu`, `sched_getaffinity` and `getrlimit`. A file that
cannot be read gives None, and the plan then takes no limit from it. Inside a container only the container's own
cgroup is visible, so a limit set on a parent cgroup is not read.
"""

from __future__ import annotations

import math
import os
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

    @property
    def memory_limit(self) -> int | None:
        """The most host memory the engine can take: the smaller of `MemAvailable` and `memory.max`."""
        known = [v for v in (self.mem_available, self.cgroup_memory_max) if v is not None]
        return min(known) if known else None

    @property
    def usable_cpus(self) -> int:
        """CPUs the process can keep busy at once: its affinity, cut to the cgroup's quota."""
        if self.cpu_quota is None:
            return self.cpus
        return max(1, min(self.cpus, math.ceil(self.cpu_quota)))


@dataclass(frozen=True, slots=True)
class Card:
    """One card the engine can use, from the core's `engine_cards` (Level Zero)."""

    name: str
    pci: str
    memory: int
    """Device-local memory, bytes."""
    device_id: int
    ip_version: int
    """Level Zero's packed IP version; 0 when not reported."""
    xe_cores: int
    pcie_gen: int
    """The link's most, as the driver reports it; -1: unknown."""
    pcie_width: int
    pcie_bytes_per_s: int

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
        cgroup_memory_max=_limit(cg / "memory.max") if cg else None,
        cgroup_swap_max=_limit(cg / "memory.swap.max") if cg else None,
        memlock=None if soft == resource.RLIM_INFINITY else soft,
        cpus=len(affinity),
        cpu_quota=cpu_quota(cg / "cpu.max") if cg else None,
        physical_cores=physical_cores(affinity, root / "sys/devices/system/cpu"),
        cpu_flags=flags,
        cpu_model=model,
    )
