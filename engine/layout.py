"""The layout the cards run, chosen at load by the step-time model, and its prediction set against the steps it ran
(DESIGN.md, section 5).

`--layout` names a layout or asks for `auto`, which costs every layout the constraints allow at the rows a step had
in the engine's last life, and takes the cheapest the core runs. The core runs one group of every card today, so
`auto` takes it and the costs are logged; a layout the core does not run is refused. Every `STEP_RING` steps the
engine process logs the predicted step against the measured one at each row count.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from engine.api.interface import STEP_RING, StepRecord
from engine.memory_plan import ModelMemory, PlanOptions
from engine.step_time import (
    Layout,
    LayoutError,
    ModelSteps,
    Terms,
    counted_miss_wait_ms,
    layout,
    layouts,
    step_time,
)

LAYOUTS = "auto, tensor, layers or groups:<g>x<w>"

_GROUPS = re.compile(r"^groups:(?P<g>\d+)x(?P<w>\d+)$")


def parse(spec: str, cards: int) -> tuple[int, int] | None:
    """`--layout` as (groups, width), or None for auto. `tensor` is one group of every card, `layers` a group a
    card. Raises ValueError for anything else."""
    match spec:
        case "auto":
            return None
        case "tensor":
            return 1, cards
        case "layers":
            return cards, 1
        case _:
            m = _GROUPS.match(spec)
            if m is None:
                raise ValueError(f"--layout {spec!r}: give {LAYOUTS}")
            return int(m.group("g")), int(m.group("w"))


def histogram(steps: Iterable[StepRecord]) -> dict[int, int]:
    """Steps by their rows."""
    return dict(sorted(Counter(s.rows for s in steps).items()))


def read_histogram(path: Path) -> dict[int, int]:
    """The rows of the steps a steps file holds (the engine process's `steps.json`); empty when there is none or it
    does not parse."""
    try:
        steps = json.loads(path.read_text())["steps"]
        return histogram(StepRecord(**s) for s in steps)
    except OSError, ValueError, KeyError, TypeError:
        return {}


@dataclass(frozen=True, slots=True)
class Choice:
    layout: Layout
    rows: Mapping[int, int]
    """The histogram the layouts were costed at: steps by rows."""
    costs: Mapping[str, float]
    """Each allowed layout's time a token over the histogram, ms."""

    def line(self) -> str:
        rows = ", ".join(f"{n} of {r}" for r, n in self.rows.items())
        costs = ", ".join(f"{k} {v:.2f} ms" for k, v in self.costs.items())
        cached = self.layout.experts_cached
        return f"layout {self.layout}, {cached} experts cached a layer; a token: {costs}; steps: {rows}"


def choose(
    spec: str,
    cards: int,
    model: ModelSteps,
    terms: Terms,
    *,
    memory: ModelMemory,
    options: PlanOptions,
    rows: Mapping[int, int],
    runs: Collection[tuple[int, int]],
) -> Choice:
    """The layout `spec` asks for, costed over `rows` (one row when empty). `runs`: the (groups, width) the core
    runs. Raises LayoutError for a named layout that breaks a constraint or that the core does not run, and when no
    layout the core runs is allowed; ValueError for a spec that does not parse."""
    hist = dict(rows) or {1: 1}
    total = sum(hist.values())

    def cost(lay: Layout) -> float:
        return sum(n * step_time(terms, model, lay, r).token_ms for r, n in hist.items()) / total

    allowed = layouts(cards, model, terms, memory=memory, options=options)
    costs = {str(lay): cost(lay) for lay in allowed}
    named = parse(spec, cards)
    if named is None:
        candidates = [lay for lay in allowed if (lay.groups, lay.width) in runs]
        if not candidates:
            raise LayoutError(f"no layout the core runs is allowed on {cards} cards")
        return Choice(min(candidates, key=cost), hist, costs)
    lay = layout(*named, cards, model, terms, memory=memory, options=options)
    if named not in runs:
        raise LayoutError(f"the core does not run {lay} yet")
    return Choice(lay, hist, costs or {str(lay): cost(lay)})


class StepLog:
    """The prediction against the steps: every `every` steps, at each row count among them, the predicted step and
    CPU wait beside the measured medians. The prediction takes M from the steps' own miss counts, as the model does
    once steps have run, and gives the hit-rate curve's beside it."""

    def __init__(
        self, terms: Terms, model: ModelSteps, lay: Layout, *, experts_cached: int | None = None, every: int = STEP_RING
    ) -> None:
        """experts_cached: the engine's, in place of the layout's plan."""
        self.terms, self.model, self.layout, self.every = terms, model, lay, every
        self.experts_cached = experts_cached
        self._last: StepRecord | None = None
        self._new: list[StepRecord] = []

    def feed(self, steps: Sequence[StepRecord]) -> list[str]:
        """The engine's last steps, oldest first, as `Engine.steps` gives them; the lines to log, if `every` new
        steps have come since the last lines."""
        start = 0
        for i in range(len(steps) - 1, -1, -1):
            if steps[i] is self._last:
                start = i + 1
                break
        if steps:
            self._last = steps[-1]
        self._new.extend(steps[start:])
        if len(self._new) < self.every:
            return []
        new, self._new = self._new, []
        lines: list[str] = []
        for rows in sorted({s.rows for s in new}):
            these = [s for s in new if s.rows == rows]
            p = step_time(self.terms, self.model, self.layout, rows, experts_cached=self.experts_cached)
            waits = [counted_miss_wait_ms(self.terms, self.model, rows, s.layers_missed, s.misses) for s in these]
            m = statistics.median(waits)
            predicted = m + p.step_ms - p.miss_ms
            step_ms = statistics.median(s.ns for s in these) / 1e6
            off = 100 * (step_ms / predicted - 1)
            cards = statistics.median(s.card0_ns for s in these) / 1e6
            wait = statistics.median(s.cpu_wait_ns for s in these) / 1e6
            plural = "s" if rows > 1 else ""
            lines.append(
                f"{self.layout}, {rows} row{plural}, {len(these)} steps: predicted {predicted:.2f} ms at their misses "
                f"({p.step_ms:.2f} at the hit-rate curve's), measured {step_ms:.2f} ({off:+.1f}%); "
                f"card 0 {cards:.2f}; CPU wait predicted {m:.2f} ({p.miss_ms:.2f}), measured {wait:.2f}"
            )
        return lines
