"""Reading the timing records, and the scheduler's cost model (DESIGN.md, section 7; ADR-0017).

The core records, the control plane reports: records are drained with `read_timing` and summed here. The cost model
is what ADR-0017's time slices need: the measured time of a decode step, and of a prompt chunk as a fixed part and a
part per row.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from engine.api.interface import WINDOW_ROWS, TimingRecords, TimingStage


@dataclass(slots=True)
class Ewma:
    value: float
    weight: float = 0.2
    samples: int = 0

    def add(self, x: float) -> None:
        self.value = x if self.samples == 0 else self.value + self.weight * (x - self.value)
        self.samples += 1


UNFITTED_STRETCH = 1.6
"""Before the fit has seen a chunk: a prompt piece may run to this multiple of its time budget rather than be cut."""
UNFITTED_WHOLE_ROWS = 1024
"""Before the fit has seen a chunk: the most rows a prompt piece holds beside writers."""


@dataclass(slots=True)
class CostModel:
    """Starts from DESIGN.md's estimates (sections 5 and 11), then follows the measured steps. A prompt chunk's time
    is a fixed part, the staging of the layers' uncached experts, and a part per row: both fitted from the chunks
    seen, decayed, as least squares over (chunks, rows) and the prompt's share of the plan's time."""

    decode_ms: Ewma
    chunk_ms: float
    row_ms: float
    decay: float = 0.8
    _s: list[float] = field(default_factory=lambda: [0.0] * 5)
    """The decayed sums: chunks², chunks x rows, rows², chunks x ms, rows x ms."""
    chunks_seen: int = 0
    """The chunks `observe` has fitted. While it is 0, `chunk_ms` and `row_ms` are only the starting values."""

    @classmethod
    def initial(cls, decode_ms: float = 11.0, row_ms: float = 0.46, chunk_ms: float = 0.0) -> CostModel:
        return cls(Ewma(decode_ms), chunk_ms, row_ms)

    def observe(self, decoded: int, rows: int, ms: float, chunks: int = 1) -> None:
        """A plan's time: `decoded` decode rows and `rows` prompt rows in `chunks` chunks (pieces read by the prompt
        path; pieces read as windows are not fitted)."""
        if rows == 0:
            if decoded:
                self.decode_ms.add(ms)
            return
        if chunks == 0:
            return
        self.chunks_seen += chunks
        y = max(0.0, ms - (self.decode_ms.value if decoded else 0.0))
        s, d = self._s, self.decay
        for i, v in enumerate((chunks * chunks, chunks * rows, rows * rows, chunks * y, rows * y)):
            s[i] = d * s[i] + v
        det = s[0] * s[2] - s[1] * s[1]
        if det > 0.05 * s[0] * s[2]:  # pieces whose rows differ enough to tell the parts apart
            a = (s[3] * s[2] - s[4] * s[1]) / det
            b = (s[0] * s[4] - s[1] * s[3]) / det
            if a >= 0 and b >= 0:
                self.chunk_ms, self.row_ms = a, b
                return
        # Chunks of about one size: the fixed part takes what the row part does not explain.
        self.chunk_ms = max(0.0, (s[3] - self.row_ms * s[1]) / s[0])

    def prompt_ms(self, rows: int) -> float:
        """A piece of `rows` rows read as one chunk."""
        return (self.chunk_ms if rows > WINDOW_ROWS else 0.0) + self.row_ms * rows

    def whole_ms(self, budget_ms: float) -> float:
        """The time a prompt piece may take and stay whole: the budget and one more cut's fixed part, since a cut
        pays that part again. Before the fit has seen a chunk, UNFITTED_STRETCH x the budget."""
        if self.chunks_seen == 0:
            return UNFITTED_STRETCH * budget_ms
        return budget_ms + self.chunk_ms

    def whole_rows(self, cap: int) -> int:
        """The most rows a prompt piece holds beside writers: the rows whose part equals the fixed part, at least
        two windows and at most `cap` (the rows of a chunk). Before the fit has seen a chunk, UNFITTED_WHOLE_ROWS."""
        if self.chunks_seen == 0:
            return UNFITTED_WHOLE_ROWS
        return min(cap, max(2 * WINDOW_ROWS, round(self.chunk_ms / max(self.row_ms, 1e-6))))

    def rows_within(self, budget_ms: float, wanted: int, cap: int) -> int:
        """The rows of a piece that aims at `budget_ms`, of the `wanted` rows left; `cap` is the rows of a chunk. All
        of them when they are at most `whole_rows` and their chunk costs at most `whole_ms` or its fixed part is at
        least its rows' part, since each cut would pay the fixed part again and a short prompt's chunk is mostly that
        part, the staging (docs/measurements/MEASUREMENTS.md#2026-10-17-prefill-timings). Otherwise the rows that
        fit the budget, but no fewer than make the rows' part equal the fixed one or half of `whole_rows`, whichever
        is fewer, and no more than `whole_rows`; all of them when fewer than WINDOW_ROWS would be left."""
        row_ms = max(self.row_ms, 1e-6)
        even = self.chunk_ms / row_ms  # the rows whose part equals the fixed part
        whole = self.whole_rows(cap)
        if wanted <= whole and (wanted <= even or self.prompt_ms(wanted) <= self.whole_ms(budget_ms)):
            return wanted
        fit = int((budget_ms - self.chunk_ms) / row_ms)
        rows = min(whole, max(1, fit, min(math.ceil(even), whole // 2)))
        return wanted if wanted - rows < WINDOW_ROWS else rows


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
