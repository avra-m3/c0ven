from __future__ import annotations

import ctypes
import mmap

import numpy as np
import pytest

from engine.api import abi
from engine.api.interface import TIMING_DTYPE, TimingStage
from engine.timing import WHOLE_ROWS, CostModel, summarise
from tools.attach import HEADER, RING, Reader, Region, trace_events


def test_record_is_32_bytes():
    assert TIMING_DTYPE.itemsize == 32


def test_cost_model_follows_measured_steps():
    cost = CostModel.initial(decode_ms=11.0, row_ms=0.5)
    cost.observe(decoded=1, rows=0, ms=9.0)
    assert cost.decode_ms.value == 9.0, "the first measurement replaces the estimate"
    cost.observe(decoded=1, rows=1000, ms=509.0)
    assert cost.chunk_ms == pytest.approx(0.0), "one size of chunk: the rows explain it all"
    cost.observe(decoded=1, rows=1000, ms=609.0)
    assert cost.row_ms == 0.5, "chunks of one size: the fixed part takes what the rows do not explain"
    assert cost.chunk_ms == pytest.approx((0.8 * 500 + 600 - 0.5 * 1800) / 1.8), "decayed by 0.8 a chunk"


def test_cost_model_splits_a_chunk_into_fixed_and_per_row_parts():
    cost = CostModel.initial(decode_ms=10.0, row_ms=0.46)
    for rows in (300, 2048, 300, 1024, 500, 2048):
        cost.observe(decoded=1, rows=rows, ms=10.0 + 600.0 + 0.07 * rows)
    assert cost.chunk_ms == pytest.approx(600.0, rel=1e-6)
    assert cost.row_ms == pytest.approx(0.07, rel=1e-6)
    assert cost.prompt_ms(1000) == pytest.approx(670.0)
    assert cost.prompt_ms(64) == pytest.approx(64 * 0.07), "a piece read as windows pays no staging"


def test_a_short_prompt_is_one_piece_and_a_long_one_is_cut():
    cost = CostModel.initial(row_ms=0.07, chunk_ms=600.0)
    assert cost.rows_within(500.0, 300) == 300, "a short prompt is whole, though its chunk is over the budget"
    assert cost.rows_within(500.0, WHOLE_ROWS) == WHOLE_ROWS, "the staging is still the larger part of its chunk"
    assert cost.rows_within(500.0, WHOLE_ROWS + 76) == WHOLE_ROWS // 2, "longer: cut, though the staging is most"
    dear = CostModel.initial(row_ms=0.3, chunk_ms=900.0)
    assert dear.rows_within(500.0, 2048) == WHOLE_ROWS // 2, "nothing fits the budget: half of WHOLE_ROWS"
    cheap = CostModel.initial(row_ms=0.5, chunk_ms=100.0)
    assert cheap.rows_within(500.0, 2048) == 800, "the rows that fit the budget"
    assert cheap.rows_within(500.0, 1000) == 1000, "1.6 x the budget is 800 ms, 1,400 rows"
    cheaper = CostModel.initial(row_ms=0.07, chunk_ms=100.0)
    assert cheaper.rows_within(500.0, 2048) == WHOLE_ROWS, "no more rows than WHOLE_ROWS, however cheap"
    assert cheaper.rows_within(500.0, WHOLE_ROWS + 40) == WHOLE_ROWS + 40, "fewer than a window's rows left: they join"
    per_row = CostModel.initial(row_ms=0.46)
    assert per_row.rows_within(100.0, 3000) == 217, "without a fixed part, pieces fit the budget"


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
