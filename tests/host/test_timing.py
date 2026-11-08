from __future__ import annotations

import ctypes
import mmap

import numpy as np
import pytest

from engine.api import abi
from engine.api.interface import TIMING_DTYPE, WINDOW_ROWS, TimingStage
from engine.timing import WHOLE_ROWS, WINDOW_ROWS_RANGE, CostModel, short_rows, summarise, whole_rows
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
    assert cost.rows_within(500.0, WHOLE_ROWS + 76) == WHOLE_ROWS, "longer: cut, the piece full, the staging paid once"
    dear = CostModel.initial(row_ms=0.3, chunk_ms=900.0)
    assert dear.rows_within(500.0, 2048) == WHOLE_ROWS // 2, "nothing fits the budget: half of WHOLE_ROWS"
    cheap = CostModel.initial(row_ms=0.5, chunk_ms=100.0)
    assert cheap.rows_within(500.0, 2048) == WHOLE_ROWS, "1.6 x the budget fits more than WHOLE_ROWS"
    dearer = CostModel.initial(row_ms=1.0, chunk_ms=100.0)
    assert dearer.rows_within(500.0, 2048) == 700, "the rows that fit 1.6 x the budget"
    assert cheap.rows_within(500.0, 1000) == 1000, "1.6 x the budget is 800 ms, 1,400 rows"
    cheaper = CostModel.initial(row_ms=0.07, chunk_ms=100.0)
    assert cheaper.rows_within(500.0, 2048) == WHOLE_ROWS, "no more rows than WHOLE_ROWS, however cheap"
    assert cheaper.rows_within(500.0, WHOLE_ROWS + 40) == WHOLE_ROWS + 40, "fewer than a window's rows left: they join"
    per_row = CostModel.initial(row_ms=0.46)
    assert per_row.rows_within(100.0, 3000) == 347, "without a fixed part, pieces fit 1.6 x the budget"


def test_a_piece_and_the_writers_beside_it_keep_to_the_fetch_path():
    assert whole_rows(1024) == 1021, "three writers' rows ride in the chunk"
    assert whole_rows(512) == 509
    assert whole_rows(0) == WHOLE_ROWS, "no fetch path: every chunk is staged one way"
    assert whole_rows(4096) == WHOLE_ROWS, "a longer fetch path: still no longer a piece than WHOLE_ROWS"
    cost = CostModel.initial(row_ms=0.07, chunk_ms=100.0)
    assert cost.rows_within(500.0, 2048, whole_rows(1024)) == 1021
    assert cost.rows_within(500.0, 1100, whole_rows(1024)) == 1021, "a prompt of 1,100 rows is cut"
    dear = CostModel.initial(row_ms=0.3, chunk_ms=900.0)
    assert dear.rows_within(500.0, 2048, whole_rows(1024)) == 510, "nothing fits the budget: half of the cap"


def test_chunks_past_the_fetch_path_are_priced_on_a_line_of_their_own():
    cost = CostModel.initial(decode_ms=10.0, row_ms=0.46)
    cost.short_rows = short_rows(1024)
    assert cost.short_rows == 1021
    assert short_rows(0) == 0
    for rows in (300, 1000, 500, 1021):
        cost.observe(decoded=1, rows=rows, ms=10.0 + 247.0 + 0.31 * rows)
    assert (cost.chunk_ms, cost.row_ms) == (pytest.approx(247.0), pytest.approx(0.31))
    assert cost.prompt_ms(4352) == pytest.approx(247.0 + 0.31 * 4352), "no long chunk seen: priced as the others"
    for rows in (4352, 2048, 4352, 3072, 1300):
        cost.observe(decoded=1, rows=rows, ms=10.0 + 316.0 + 0.35 * rows)
    assert (cost.long_chunk_ms, cost.long_row_ms) == (pytest.approx(316.0), pytest.approx(0.35))
    assert (cost.chunk_ms, cost.row_ms) == (pytest.approx(247.0), pytest.approx(0.31)), "the fetch path's line stays"
    assert cost.prompt_ms(1021) == pytest.approx(247.0 + 0.31 * 1021)
    assert cost.prompt_ms(1022) == pytest.approx(316.0 + 0.35 * 1022)
    cost.observe(decoded=0, rows=2 * 4352, ms=2 * (316.0 + 0.35 * 4352), chunks=2)
    assert cost.long_chunk_ms == pytest.approx(316.0), "a plan of two whole chunks"


def test_long_chunks_of_one_size_are_priced_right_at_that_size():
    for chunk_ms in (300.0, 0.0):  # the second costs less a row than the fetch path's line
        cost = CostModel.initial(row_ms=0.378, chunk_ms=233.0)
        cost.short_rows = 1021
        for _ in range(5):
            cost.observe(decoded=0, rows=4352, ms=chunk_ms + 0.3 * 4352)
        assert cost.prompt_ms(4352) == pytest.approx(chunk_ms + 0.3 * 4352)
        assert cost.long_row_ms <= 0.378


def test_a_budget_that_fits_a_whole_chunk_reads_past_the_fetch_path():
    cost = CostModel.initial(row_ms=0.35, chunk_ms=316.0)  # a whole chunk 1,839 ms
    whole = whole_rows(1024)
    for budget in (500.0, 1100.0):
        assert not cost.whole_chunks(budget, 4352)
        assert cost.rows_within(budget, 5000, whole, None, 4352) == cost.rows_within(budget, 5000, whole), "as before"
    assert cost.whole_chunks(1150.0, 4352)
    assert cost.rows_within(1150.0, 4352, whole, None, 4352) == 4352, "a whole chunk"
    assert cost.rows_within(2000.0, 2600, whole, 900.0, 4352) == 2600, "the prompt's rest, whatever the aim"
    assert cost.rows_within(2000.0, 1100, whole, None, 4352) == 1100
    assert cost.rows_within(2000.0, 600, whole, None, 4352) == 600, "a short prompt is whole, as before"
    assert cost.rows_within(2000.0, 4352, whole) == whole, "no chunk_rows: kept to whole_rows"


def test_a_piece_aims_by_the_fixed_part_and_what_decode_is_worth():
    full = CostModel.initial(decode_ms=21.0, row_ms=0.4, chunk_ms=200.0)
    assert full.piece_ms(500.0, 0.5, 30, 2) == 800.0, "a large fixed part: the most the stretch allows"
    assert full.rows_within(500.0, 2048, 1021, 800.0) == 1021
    small = CostModel.initial(decode_ms=21.0, row_ms=2.0, chunk_ms=50.0)
    assert small.piece_ms(500.0, 0.5, 30, 2) == 500.0, "a small fixed part: more time returns little, so the budget"
    middle = CostModel.initial(decode_ms=21.0, row_ms=1.0, chunk_ms=160.0)
    aim = middle.piece_ms(500.0, 0.5, 30, 2)
    assert 500.0 < aim < 800.0
    assert middle.piece_ms(500.0, 0.5, 0, 2) == 800.0, "with decode not counted, longer pieces return more"
    assert middle.piece_ms(500.0, 0.5, 60, 2) < aim, "worth more, decode stops a piece sooner"


def test_chunks_of_one_size_keep_the_split_from_earlier_sizes():
    cost = CostModel.initial(decode_ms=10.0, row_ms=0.46)
    for rows in (300, 1000, 300, 1000, 500, 1000):
        cost.observe(decoded=0, rows=rows, ms=250.0 + 0.3 * rows)
    assert cost.row_ms == pytest.approx(0.3, rel=1e-6)
    for _ in range(40):
        cost.observe(decoded=0, rows=1021, ms=270.0 + 0.3 * 1021)
    assert cost.row_ms == pytest.approx(0.3, abs=0.05), "one size says little new about the time per row"
    assert cost.prompt_ms(1021) == pytest.approx(270.0 + 0.3 * 1021, rel=0.01), "the chunk's time follows recent chunks"


def test_window_rows_is_where_a_chunk_costs_less_than_windows():
    cost = CostModel.initial(decode_ms=10.0, row_ms=0.46, chunk_ms=466.0)
    assert cost.window_rows() == WINDOW_ROWS, "no window plan seen yet"
    cost.observe(decoded=0, rows=64, ms=330.5, chunks=0)
    assert cost.window_row_ms.value == pytest.approx(330.5 / 64)
    assert cost.window_rows() == int(466.0 / (330.5 / 64 - 0.46))
    cost.observe(decoded=1, rows=16, ms=10.0 + 16 * 5.0, chunks=0)
    assert cost.window_row_ms.value == pytest.approx(0.8 * 330.5 / 64 + 0.2 * 5.0), "decode time is taken off"
    assert cost.window_row_ms.samples == 2


def test_window_rows_needs_a_chunk_and_stays_in_range():
    no_chunk = CostModel.initial(row_ms=0.46)
    no_chunk.observe(decoded=0, rows=64, ms=330.5, chunks=0)
    assert no_chunk.window_rows() == WINDOW_ROWS, "no chunk fitted: the fixed part is unknown"
    low, high = WINDOW_ROWS_RANGE
    dear = CostModel.initial(row_ms=0.46, chunk_ms=10.0)
    dear.observe(decoded=0, rows=10, ms=50.0, chunks=0)
    assert dear.window_rows() == low
    cheap = CostModel.initial(row_ms=0.46, chunk_ms=5000.0)
    cheap.observe(decoded=0, rows=10, ms=10.0, chunks=0)
    assert cheap.window_rows() == high, "windows no dearer per row than a chunk: always windows"


def test_the_threshold_decides_which_pieces_pay_the_fixed_part():
    cost = CostModel.initial(row_ms=0.07, chunk_ms=100.0)
    cost.observe(decoded=0, rows=10, ms=10.7, chunks=0)
    assert cost.window_rows() == 100
    assert cost.prompt_ms(100) == pytest.approx(100 * 0.07)
    assert cost.prompt_ms(101) == pytest.approx(100.0 + 101 * 0.07)
    assert cost.rows_within(500.0, WHOLE_ROWS + 99) == WHOLE_ROWS + 99, "fewer than window_rows() left: they join"
    assert cost.rows_within(500.0, WHOLE_ROWS + 100) == WHOLE_ROWS


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


def test_the_cost_model_carries_from_one_run_to_the_next():
    cost = CostModel.initial(decode_ms=10.0, row_ms=0.46)
    for rows in (300, 2048, 300, 1024):
        cost.observe(decoded=1, rows=rows, ms=10.0 + 600.0 + 0.07 * rows)
    cost.observe(decoded=0, rows=64, ms=330.5, chunks=0)
    cost.short_rows = 1021
    cost.observe(decoded=0, rows=4352, ms=1839.0)
    again = CostModel.from_dict(cost.to_dict())
    again.short_rows = 1021
    assert again.prompt_ms(4352) == pytest.approx(1839.0), "the long line carries"
    assert again.to_dict() == cost.to_dict()
    assert again.window_rows() == cost.window_rows(), "the window plans seen carry too"
    assert again.decode_ms.samples == 0, "the decode time is a start value: the first step replaces it"
    again.observe(decoded=1, rows=0, ms=9.0)
    assert again.decode_ms.value == 9.0


def test_a_cost_model_that_is_not_one_is_refused():
    good = CostModel.initial().to_dict()
    with pytest.raises(ValueError, match="wrong size"):
        CostModel.from_dict({**good, "s": [0.0] * 4})
    with pytest.raises(ValueError, match="finite"):
        CostModel.from_dict({**good, "row_ms": float("nan")})
    with pytest.raises(KeyError):
        CostModel.from_dict({k: v for k, v in good.items() if k != "window_row_ms"})
    with pytest.raises(KeyError):
        CostModel.from_dict({k: v for k, v in good.items() if k != "s_long"})
    with pytest.raises(ValueError, match="wrong size"):
        CostModel.from_dict({**good, "s_long_slow": [0.0] * 6})
