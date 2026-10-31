"""The host probe against a made-up /proc and /sys."""

from __future__ import annotations

from pathlib import Path

from engine.host import GIB, Card, read


def make_root(tmp_path: Path, cpu_max: str = "max 100000", memory_max: str = "max", smt: bool = True) -> Path:
    (tmp_path / "proc/self").mkdir(parents=True)
    (tmp_path / "proc/meminfo").write_text(
        "MemTotal:       94435076 kB\nMemFree:  1000 kB\nMemAvailable:   58829088 kB\nSwapTotal:      94371836 kB\n"
    )
    (tmp_path / "proc/self/cgroup").write_text("0::/box\n")
    (tmp_path / "proc/cpuinfo").write_text(
        "processor\t: 0\nmodel name\t: AMD Ryzen 9 7945HX\nflags\t\t: fpu avx2 avx512f avx512_vnni\n\n"
        "processor\t: 1\nmodel name\t: AMD Ryzen 9 7945HX\nflags\t\t: fpu\n"
    )
    cg = tmp_path / "sys/fs/cgroup/box"
    cg.mkdir(parents=True)
    (cg / "cpu.max").write_text(cpu_max + "\n")
    (cg / "memory.max").write_text(memory_max + "\n")
    (cg / "memory.swap.max").write_text("0\n")
    for c in range(32):
        topo = tmp_path / f"sys/devices/system/cpu/cpu{c}/topology"
        topo.mkdir(parents=True)
        (topo / "physical_package_id").write_text("0\n")
        (topo / "core_id").write_text(f"{c % 16 if smt else c}\n")
    return tmp_path


def test_memory_and_its_limits(tmp_path: Path):
    h = read(make_root(tmp_path, memory_max=str(60 * GIB)), affinity=set(range(32)))
    assert h.mem_available == 58829088 * 1024
    assert h.cgroup_memory_max == 60 * GIB
    assert h.cgroup_swap_max == 0
    assert h.memory_limit == 58829088 * 1024


def test_the_cgroup_limit_wins_when_smaller(tmp_path: Path):
    h = read(make_root(tmp_path, memory_max=str(40 * GIB)), affinity=set(range(32)))
    assert h.memory_limit == 40 * GIB


def test_cpus_cores_and_quota(tmp_path: Path):
    h = read(make_root(tmp_path, cpu_max="1600000 100000"), affinity=set(range(32)))
    assert (h.cpus, h.physical_cores, h.cpu_quota, h.usable_cpus) == (32, 16, 16.0, 16)
    assert "avx512_vnni" in h.cpu_flags
    assert h.cpu_model == "AMD Ryzen 9 7945HX"


def test_no_quota_and_an_affinity_on_one_thread_of_each_core(tmp_path: Path):
    h = read(make_root(tmp_path), affinity=set(range(8)))
    assert (h.cpus, h.physical_cores, h.cpu_quota, h.usable_cpus) == (8, 8, None, 8)


def test_nothing_readable_gives_no_limits(tmp_path: Path):
    h = read(tmp_path, affinity={0, 1})
    assert (h.mem_available, h.cgroup_memory_max, h.memory_limit, h.cpu_quota) == (None, None, None, None)
    assert h.physical_cores == 2


def test_a_cards_ip_version_and_link():
    card = Card(
        "Intel(R) Arc(TM) Pro B60 Graphics",
        "0000:03:00.0",
        24_385_683_456,
        0xE211,
        (20 << 22) | (1 << 14),
        20,
        5,
        8,
        -1,
    )
    assert card.ip == "20.1.0"
    assert card.link == "PCIe 5.0 x8"
    assert Card("x", "", 0, 0, 0, 0, -1, -1, -1).ip == "unknown"


def test_workers_one_per_core_the_quota_allows(tmp_path: Path):
    assert read(make_root(tmp_path), affinity=set(range(32))).cpu_workers() == (16, "cores")


def test_workers_under_a_quota(tmp_path: Path):
    assert read(make_root(tmp_path, cpu_max="800000 100000"), affinity=set(range(32))).cpu_workers() == (8, "cpu.max")
    h = read(make_root(tmp_path / "half", cpu_max="350000 100000"), affinity=set(range(32)))
    assert h.cpu_workers() == (4, "cpu.max")


def test_workers_on_one_thread_of_eight_cores(tmp_path: Path):
    assert read(make_root(tmp_path), affinity=set(range(8))).cpu_workers() == (8, "cores")
