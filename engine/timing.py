"""Reading the timing records, and the scheduler's cost model (DESIGN.md, section 7; ADR-0017).

The core records, the control plane reports: records are drained with `read_timing` and summed here. The cost model
is what ADR-0017's time slices need: the measured time of a decode step and of one prompt row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from engine.api.interface import TimingRecords, TimingStage


@dataclass(slots=True)
class Ewma:
    value: float
    weight: float = 0.2
    samples: int = 0

    def add(self, x: float) -> None:
        self.value = x if self.samples == 0 else self.value + self.weight * (x - self.value)
        self.samples += 1


@dataclass(slots=True)
class CostModel:
    """Starts from DESIGN.md's estimates (sections 5 and 11), then follows the measured steps."""

    decode_ms: Ewma
    row_ms: Ewma

    @classmethod
    def initial(cls, decode_ms: float = 11.0, row_ms: float = 0.46) -> CostModel:
        return cls(Ewma(decode_ms), Ewma(row_ms))

    def observe(self, decoded: int, rows: int, ms: float) -> None:
        if rows == 0:
            if decoded:
                self.decode_ms.add(ms)
            return
        prompt_ms = max(0.0, ms - (self.decode_ms.value if decoded else 0.0))
        self.row_ms.add(prompt_ms / rows)

    def rows_within(self, budget_ms: float) -> int:
        return max(1, int(budget_ms / max(self.row_ms.value, 1e-6)))


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
