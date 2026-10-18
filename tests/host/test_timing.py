from __future__ import annotations

import ctypes
import mmap

import numpy as np
import pytest

from engine.api import abi
from engine.api.interface import TIMING_DTYPE, TimingStage
from engine.timing import CostModel, summarise
from tools.attach import HEADER, RING, Reader, Region, trace_events


def test_record_is_32_bytes():
    assert TIMING_DTYPE.itemsize == 32


def test_cost_model_follows_measured_steps():
    cost = CostModel.initial(decode_ms=11.0, row_ms=0.5)
    cost.observe(decoded=1, rows=0, ms=9.0)
    assert cost.decode_ms.value == 9.0, "the first measurement replaces the estimate"
    cost.observe(decoded=1, rows=1000, ms=409.0)
    assert cost.row_ms.value == 0.4
    assert cost.rows_within(100.0) == 250


def test_summarise():
    records = np.zeros(4, dtype=TIMING_DTYPE)
    records["stage"] = [TimingStage.STEP, TimingStage.STEP, TimingStage.CPU_MISS, TimingStage.STEP]
    records["duration"] = [10_000_000, 12_000_000, 90_000, 11_000_000]
    by_stage = {s.stage: s for s in summarise(records)}
    assert by_stage[TimingStage.STEP].records == 3
    assert by_stage[TimingStage.STEP].total_ms == 33.0
    assert by_stage[TimingStage.STEP].median_us == 11_000.0
    assert by_stage[TimingStage.CPU_MISS].median_us == 90.0


def make_region(rings: int, records: int) -> mmap.mmap:
    size = HEADER + rings * (RING + records * TIMING_DTYPE.itemsize)
    buffer = mmap.mmap(-1, size)
    header = abi.CTimingRegion.from_buffer(buffer)
    header.magic = abi.TIMING_MAGIC
    header.abi_version = abi.TIMING_REGION_VERSION
    header.n_rings = rings
    header.ring_records = records
    del header
    return buffer


def write(region: Region, ring: int, step: int) -> None:
    """What the core's writer does: the record, then the head."""
    r = region.rings[ring]
    size = region.header.ring_records
    region.records[ring][r.head % size] = (step * 1000, 500, step, TimingStage.STEP, ring, 0, 0)
    r.head += 1


def test_attach_reads_rings_across_the_wrap_and_counts_losses():
    buffer = make_region(rings=2, records=4)
    region = Region.over(buffer)
    region.rings[1].thread = 7
    reader = Reader.at_head(region)
    for step in range(3):
        write(region, 0, step)
    assert [list(r["step"]) for r in reader.read()] == [[0, 1, 2], []]
    for step in range(3, 6):
        write(region, 0, step)
    write(region, 1, 9)
    first, second = reader.read()
    assert list(first["step"]) == [3, 4, 5]
    assert list(second["step"]) == [9]
    for step in range(6, 12):
        write(region, 0, step)
    assert list(reader.read()[0]["step"]) == [8, 9, 10, 11]
    assert reader.lost == 2
    region.set_detail(1)
    assert abi.CTimingRegion.from_buffer(buffer).detail == 1
    events = trace_events([second], [7])
    assert events[0]["name"] == "step"
    assert events[0]["tid"] == 7
    assert events[0]["dur"] == 0.5


def test_attach_refuses_other_memory():
    buffer = mmap.mmap(-1, 4096)
    with pytest.raises(ValueError, match="not an engine timing region"):
        Region.over(buffer)
    assert ctypes.sizeof(abi.CTimingRegion) == HEADER
