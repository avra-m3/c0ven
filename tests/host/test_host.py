"""The host probe against a made-up /proc and /sys."""

from __future__ import annotations

from pathlib import Path

from engine.host import GIB, Card, read, slot_link


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


def test_the_free_reserved_huge_pages(tmp_path: Path):
    root = make_root(tmp_path)
    (root / "proc/meminfo").write_text(
        "MemAvailable: 1000 kB\nHugePages_Total: 40\nHugePages_Free: 30\nHugePages_Rsvd: 4\nHugepagesize: 2048 kB\n"
    )
    assert read(root, affinity={0}).huge_pages_free == 26 * 2 * 2**20


def test_no_huge_pages_reported_gives_none(tmp_path: Path):
    assert read(make_root(tmp_path), affinity={0}).huge_pages_free is None


def b65(pcie_gen: int = 5, pcie_width: int = 16) -> Card:
    return Card("Intel(R) Arc(TM) Pro B65 Graphics", "0000:19:00.0", 32 * GIB, 0xE212, 0, 20, pcie_gen, pcie_width, -1)


def make_pci(root: Path, chain: list[tuple[str, str | None, str | None]]) -> None:
    """A device path under /sys/devices from the root port down, each (address, max_link_speed, current_link_width),
    with /sys/bus/pci/devices linking the last."""
    d = root / "sys/devices/pci0000:00"
    for bdf, speed, width in chain:
        d = d / bdf
        d.mkdir(parents=True)
        if speed is not None:
            (d / "max_link_speed").write_text(f"{speed} PCIe\n")
        if width is not None:
            (d / "current_link_width").write_text(f"{width}\n")
    (root / "sys/bus/pci/devices").mkdir(parents=True)
    (root / "sys/bus/pci/devices" / chain[-1][0]).symlink_to(d)


def test_a_card_on_a_root_port_keeps_its_link(tmp_path: Path):
    make_pci(tmp_path, [("0000:00:01.1", "32.0 GT/s", "16"), ("0000:19:00.0", "32.0 GT/s", "16")])
    card = slot_link(b65(), tmp_path)
    assert card.link == "PCIe 5.0 x16"
    assert card.pcie_bytes_per_s == round(32e9 / 8 * 128 / 130 * 16)


def test_a_card_behind_its_switch_gives_the_slots_link(tmp_path: Path):
    # The slot trained at 3.0 x8; the switch's link to the card runs at 5.0 x16.
    make_pci(
        tmp_path,
        [
            ("0000:00:01.1", "8.0 GT/s", "8"),
            ("0000:17:00.0", "32.0 GT/s", "8"),
            ("0000:18:01.0", "32.0 GT/s", "16"),
            ("0000:19:00.0", "32.0 GT/s", "16"),
        ],
    )
    assert slot_link(b65(), tmp_path).link == "PCIe 3.0 x8"


def test_no_sysfs_keeps_level_zeros_link(tmp_path: Path):
    assert slot_link(b65(), tmp_path) == b65()


def test_missing_link_files_keep_level_zeros_value(tmp_path: Path):
    make_pci(tmp_path, [("0000:00:01.1", None, "8"), ("0000:19:00.0", "Unknown", None)])
    assert slot_link(b65(), tmp_path).link == "PCIe 5.0 x8"
    make_pci(tmp_path / "none", [("0000:00:01.1", None, None), ("0000:19:00.0", None, None)])
    assert slot_link(b65(), tmp_path / "none") == b65()
