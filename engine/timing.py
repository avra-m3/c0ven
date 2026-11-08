"""Reading the timing records, and the scheduler's cost model (DESIGN.md, section 7; ADR-0017).

The core records, the control plane reports: records are drained with `read_timing` and summed here. The cost model
is what ADR-0017's time slices need: the measured time of a decode step, and of a prompt chunk as a fixed part and a
part per row, for each of the two ways the core stages a chunk's experts (ADR-0067).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.api.interface import PROMPT_RUNS, WINDOW_ROWS, TimingRecords, TimingStage


@dataclass(slots=True)
class Ewma:
    value: float
    weight: float = 0.2
    samples: int = 0

    def add(self, x: float) -> None:
        self.value = x if self.samples == 0 else self.value + self.weight * (x - self.value)
        self.samples += 1


STRETCH = 1.6
"""Beside writers, a long prompt's piece is sized to this multiple of the budget, and a plan's prompt pieces may be
priced at up to it before a piece is cut or another prompt stops joining, since each cut pays the fixed part again. A
piece that fills `whole_rows` reads the most tokens a second and costs the writers no tokens a second
(docs/measurements/MEASUREMENTS.md#2026-11-01-full-prompt-pieces). A larger stretch joined more prompts for no more
tokens a second (docs/measurements/MEASUREMENTS.md#2026-11-01-whole-rows-and-stretch)."""
WHOLE_ROWS = 1024
"""Beside writers, the most rows a prompt piece holds when the core has no fetch path (`whole_rows`), unless the
budget fits a whole chunk (`CostModel.whole_chunks`): a prompt of up to this many rows may be read whole, a longer one
is cut, since the fixed part is most of a chunk only up to about this many rows and a longer chunk holds the writers
for longer (docs/measurements/MEASUREMENTS.md#2026-10-17-prefill-timings)."""
PIECE_RETURN = 0.15
"""Beside writers, a piece stops growing past the budget when 1% more time returns this many percent more score, the
score being prompt rows plus `decode_weight` x decode tokens a second. It is set so that a piece on the fetch path still
fills `whole_rows` when the cost model's fixed part reads low
(docs/measurements/MEASUREMENTS.md#2026-11-01-piece-rule-with-a-slow-split)."""
SPLIT_DECAY = 0.98
"""How slowly the time per row forgets a chunk. Full pieces are all of one size and cannot tell the fixed part from the
part per row, so the split needs a longer memory than the chunk's level, which follows the position in the prompt."""
WINDOW_ROWS_RANGE = (8, 256)
"""The fewest and most rows CostModel.window_rows returns."""


def short_rows(fetch_rows: int) -> int:
    """The most prompt rows a chunk holds on the fetch path, for the core's `fetch_rows`, leaving room for the writers'
    rows that ride in it (at most PROMPT_RUNS - 1); 0 when the core stages every chunk one way."""
    return max(0, fetch_rows - (PROMPT_RUNS - 1))


def whole_rows(fetch_rows: int) -> int:
    """Beside writers, the most rows a prompt piece holds, for the core's `fetch_rows`, unless the budget fits a whole
    chunk: WHOLE_ROWS, or fewer so that the chunk, with the writers' rows that ride in it, keeps to the fetch path,
    which fetches only the experts its router picks. A longer chunk copies every uncached expert: it holds the writers
    longer and reads no more rows a second until it is far longer
    (docs/measurements/MEASUREMENTS.md#2026-11-01-whole-rows-and-stretch)."""
    return min(WHOLE_ROWS, short_rows(fetch_rows) or WHOLE_ROWS)


def _fit(s: list[float], slow: list[float], decay: float, x: tuple[float, ...], row_ms: float) -> tuple[float, float]:
    """Adds a plan's (chunks², chunks x rows, rows², chunks x ms, rows x ms) to the sums `s`, decayed by `decay`, and
    `slow`, decayed by SPLIT_DECAY. Returns the fixed part and the part per row: the part per row from `slow` when its
    chunks differ enough in size to tell the parts apart, else `row_ms`; the fixed part what the rows do not explain in
    the recent chunks."""
    for i, v in enumerate(x):
        s[i] = decay * s[i] + v
        slow[i] = SPLIT_DECAY * slow[i] + v
    det = slow[0] * slow[2] - slow[1] * slow[1]
    if det > 0.05 * slow[0] * slow[2]:
        b = (slow[0] * slow[4] - slow[1] * slow[3]) / det
        if b >= 0:
            row_ms = b
    return max(0.0, (s[3] - row_ms * s[1]) / s[0]), row_ms


@dataclass(slots=True)
class CostModel:
    """Starts from DESIGN.md's estimates (sections 5 and 11), then follows the measured steps. A prompt chunk's time
    is a fixed part, the staging of the layers' uncached experts, and a part per row: both fitted from the chunks
    seen, decayed, as least squares over (chunks, rows) and the prompt's share of the plan's time. A chunk of more
    than `short_rows` rows copies every uncached expert, so it has a line of its own, `long_chunk_ms` and
    `long_row_ms` (ADR-0067). A row read as a decode window costs `window_row_ms`, an average of the plans whose prompt
    rows were all read so."""

    decode_ms: Ewma
    chunk_ms: float
    row_ms: float
    decay: float = 0.8
    _s: list[float] = field(default_factory=lambda: [0.0] * 5)
    """The decayed sums: chunks², chunks x rows, rows², chunks x ms, rows x ms."""
    window_row_ms: Ewma = field(default_factory=lambda: Ewma(0.0))
    _slow: list[float] = field(default_factory=lambda: [0.0] * 5)
    """The same sums, decayed by SPLIT_DECAY: what `row_ms` is fitted from."""
    long_chunk_ms: float = 0.0
    long_row_ms: float = 0.0
    _long: list[float] = field(default_factory=lambda: [0.0] * 5)
    _long_slow: list[float] = field(default_factory=lambda: [0.0] * 5)
    """The two sums for chunks of more than `short_rows` rows. Until one is seen, they are priced as shorter ones."""
    short_rows: int = 0
    """The most rows a chunk has on the fetch path (`short_rows` of the core's `fetch_rows`); 0: one line for every
    chunk. The scheduler sets it. It is not kept with the rest: the kept file's key holds the model and options."""

    @classmethod
    def initial(cls, decode_ms: float = 11.0, row_ms: float = 0.46, chunk_ms: float = 0.0) -> CostModel:
        return cls(Ewma(decode_ms), chunk_ms, row_ms)

    def to_dict(self) -> dict[str, float | int | list[float]]:
        """The parts that carry from one run to the next, as JSON values."""
        return {
            "decode_ms": self.decode_ms.value,
            "chunk_ms": self.chunk_ms,
            "row_ms": self.row_ms,
            "s": list(self._s),
            "s_slow": list(self._slow),
            "window_row_ms": self.window_row_ms.value,
            "window_samples": self.window_row_ms.samples,
            "long_chunk_ms": self.long_chunk_ms,
            "long_row_ms": self.long_row_ms,
            "s_long": list(self._long),
            "s_long_slow": list(self._long_slow),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CostModel:
        """A model from `to_dict`'s output. Raises ValueError, KeyError or TypeError when it is not that. The decode
        time is a start value: the first step replaces it."""
        model = cls.initial(float(data["decode_ms"]), float(data["row_ms"]), float(data["chunk_ms"]))
        model._s = [float(x) for x in data["s"]]
        model._slow = [float(x) for x in data["s_slow"]]
        model.window_row_ms = Ewma(float(data["window_row_ms"]), samples=int(data["window_samples"]))
        model.long_chunk_ms = float(data["long_chunk_ms"])
        model.long_row_ms = float(data["long_row_ms"])
        model._long = [float(x) for x in data["s_long"]]
        model._long_slow = [float(x) for x in data["s_long_slow"]]
        sums = (model._s, model._slow, model._long, model._long_slow)
        values = [
            model.decode_ms.value,
            model.chunk_ms,
            model.row_ms,
            model.window_row_ms.value,
            model.long_chunk_ms,
            model.long_row_ms,
            *(v for s in sums for v in s),
        ]
        sizes_ok = all(len(s) == 5 for s in sums) and model.window_row_ms.samples >= 0
        if not sizes_ok or not all(math.isfinite(v) and v >= 0 for v in values):
            raise ValueError("the cost model has a wrong size or a value that is not a finite, non-negative number")
        return model

    def observe(self, decoded: int, rows: int, ms: float, chunks: int = 1) -> None:
        """A plan's time: `decoded` decode rows and `rows` prompt rows in `chunks` chunks (pieces read by the prompt
        path; the rows of a plan with none are all read as windows, and give `window_row_ms`)."""
        if rows == 0:
            if decoded:
                self.decode_ms.add(ms)
            return
        y = max(0.0, ms - (self.decode_ms.value if decoded else 0.0))
        if chunks == 0:
            self.window_row_ms.add(y / rows)
            return
        x = (chunks * chunks, chunks * rows, rows * rows, chunks * y, rows * y)
        if not self._long_chunk(rows / chunks):
            self.chunk_ms, self.row_ms = _fit(self._s, self._slow, self.decay, x, self.row_ms)
            return
        # Long chunks are mostly whole ones, of one size. Until their sizes tell the parts apart, a row costs what it
        # costs on the fetch path, which is close (docs/measurements/MEASUREMENTS.md#2026-11-01-whole-rows-and-stretch),
        # or what the line through 0 gives if that is less, so that the fixed part is not cut to 0.
        s, d = self._long, self.decay
        through_0 = (d * s[4] + rows * y) / (d * s[2] + rows * rows)
        self.long_chunk_ms, self.long_row_ms = _fit(s, self._long_slow, d, x, min(self.row_ms, through_0))

    def window_rows(self) -> int:
        """The most rows a piece may have and be read as windows: the n for which n x window_row_ms equals the chunk's
        chunk_ms + n x row_ms, within WINDOW_ROWS_RANGE. WINDOW_ROWS until a chunk and a window plan have been seen
        (docs/measurements/MEASUREMENTS.md#2026-10-17-prompt-windows)."""
        if self.chunk_ms <= 0 or self.window_row_ms.samples == 0:
            return WINDOW_ROWS
        low, high = WINDOW_ROWS_RANGE
        extra = self.window_row_ms.value - self.row_ms  # a window row's cost over a chunk row's
        return high if extra <= 0 else max(low, min(high, int(self.chunk_ms / extra)))

    def _long_chunk(self, rows: float) -> bool:
        return 0 < self.short_rows < rows

    def prompt_ms(self, rows: int) -> float:
        """A piece of `rows` rows: one chunk when it has more than window_rows() rows, priced on the long line when it
        has more than `short_rows` rows and a chunk that long has been seen."""
        if self._long_chunk(rows) and self._long[0] > 0:
            return self.long_chunk_ms + self.long_row_ms * rows
        return (self.chunk_ms if rows > self.window_rows() else 0.0) + self.row_ms * rows

    def whole_chunks(self, budget_ms: float, chunk_rows: int) -> bool:
        """Whether a prompt piece beside writers may hold a whole chunk of `chunk_rows` rows: when that chunk costs at
        most STRETCH x the budget (ADR-0067)."""
        return self.prompt_ms(chunk_rows) <= STRETCH * budget_ms

    def piece_ms(self, budget_ms: float, decode_share: float, decode_weight: int, writers: int) -> float:
        """What a long prompt's piece beside `writers` writers aims at: between the budget and STRETCH x the budget, up
        to where 1% more time returns PIECE_RETURN percent more score. A piece's rows give (time - fixed part) / row_ms
        and the decode it is owed decode_share x time / a token's time, so in units of time the score of a piece is
        time x (1 + decode_share x w) - fixed part, with w = decode_weight x row_ms / a token's time."""
        token_ms = max(self.decode_ms.value / max(writers, 1), 1e-6)
        w = decode_weight * self.row_ms / token_ms
        aim = (1 + 1 / PIECE_RETURN) * self.chunk_ms / (1 + decode_share * w)
        return min(STRETCH * budget_ms, max(budget_ms, aim))

    def rows_within(
        self, budget_ms: float, wanted: int, whole: int = WHOLE_ROWS, aim_ms: float | None = None, chunk_rows: int = 0
    ) -> int:
        """The rows of a piece that aims at `budget_ms`, of the `wanted` rows left, with `whole` the most a piece
        holds (`whole_rows`). When there are more than `whole` and the budget fits a whole chunk of `chunk_rows` rows
        (`whole_chunks`), all of them up to `chunk_rows`, since a piece past the fetch path reads more rows a second
        only as it nears a whole chunk (ADR-0067). All of them when they are at most `whole` and their chunk costs at
        most STRETCH x the budget or its fixed part is at least its rows' part, since each cut would pay the fixed part
        again and a short prompt's chunk is mostly that part, the staging
        (docs/measurements/MEASUREMENTS.md#2026-10-17-prefill-timings).
        Otherwise the rows that fit `aim_ms` (STRETCH x the budget if not given), but no fewer than make the rows' part
        equal the fixed one or half of `whole`, whichever is fewer, and no more than `whole`; all of them when fewer
        than window_rows() would be left."""
        if whole < wanted and whole < chunk_rows and self.whole_chunks(budget_ms, chunk_rows):
            return min(wanted, chunk_rows)
        row_ms = max(self.row_ms, 1e-6)
        even = self.chunk_ms / row_ms  # the rows whose part equals the fixed part
        if wanted <= whole and (wanted <= even or self.prompt_ms(wanted) <= STRETCH * budget_ms):
            return wanted
        fit = int(((STRETCH * budget_ms if aim_ms is None else aim_ms) - self.chunk_ms) / row_ms)
        rows = min(whole, max(1, fit, min(math.ceil(even), whole // 2)))
        return wanted if wanted - rows < self.window_rows() else rows


@dataclass(frozen=True, slots=True)
class StageSummary:
    stage: TimingStage
    records: int
    total_ms: float
    median_us: float
    p99_us: float


def summarise(records: TimingRecords) -> list[StageSummary]:
    """Records per stage: count, total and percentiles of their durations."""
    out: list[StageSummary] = []
    stages = records["stage"]
    durations = records["duration"].astype(np.float64)
    for stage in TimingStage:
        d = durations[stages == stage]
        if d.size == 0:
            continue
        out.append(
            StageSummary(
                stage,
                int(d.size),
                float(d.sum()) / 1e6,
                float(np.median(d)) / 1e3,
                float(np.percentile(d, 99)) / 1e3,
            )
        )
    return out
