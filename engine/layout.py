"""The layout the cards run, chosen at load by the step-time model, and its prediction set against the steps it ran
(DESIGN.md, section 5).

`--layout` names a layout or asks for `auto`, which takes the layout the core runs with the least time a token for
one request. Every layout the constraints allow is costed at `REQUESTS` requests writing, a multi-group layout with
its groups pipelined, and the costs are logged: they show where pipelining would beat one group. The core runs one
group of every card today, so `auto` takes it; a layout the core does not run is refused. Every `STEP_RING` steps
the engine process logs the predicted step against the measured one at each row count, M from the steps' own misses.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

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
    token_time,
)

LAYOUTS = "auto, tensor, layers or groups:<g>x<w>"

REQUESTS = (1, 2, 4, 8)
"""The requests writing at which the layouts are costed: one, and up to a step's most rows."""

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


@dataclass(frozen=True, slots=True)
class Choice:
    layout: Layout
    costs: Mapping[str, tuple[float, ...]]
    """Each allowed layout's time a token at each of `REQUESTS`, ms."""

    def line(self) -> str:
        costs = "; ".join(f"{k} {', '.join(f'{v:.2f}' for v in vs)} ms" for k, vs in self.costs.items())
        cached = self.layout.experts_cached
        at = ", ".join(str(r) for r in REQUESTS)
        return f"layout {self.layout}, {cached} experts cached a layer; a token at {at} requests: {costs}"


def choose(
    spec: str,
    cards: int,
    model: ModelSteps,
    terms: Terms,
    *,
    memory: ModelMemory,
    options: PlanOptions,
    runs: Collection[tuple[int, int]],
) -> Choice:
    """The layout `spec` asks for. `runs`: the (groups, width) the core runs. Raises LayoutError for a named layout
    that breaks a constraint or that the core does not run, and when no layout the core runs is allowed; ValueError
    for a spec that does not parse."""

    def cost(lay: Layout) -> tuple[float, ...]:
        return tuple(token_time(terms, model, lay, r) for r in REQUESTS)

    allowed = layouts(cards, model, terms, memory=memory, options=options)
    costs = {str(lay): cost(lay) for lay in allowed}
    named = parse(spec, cards)
    if named is None:
        candidates = [lay for lay in allowed if (lay.groups, lay.width) in runs]
        if not candidates:
            raise LayoutError(f"no layout the core runs is allowed on {cards} cards")
        return Choice(min(candidates, key=lambda lay: costs[str(lay)][0]), costs)
    lay = layout(*named, cards, model, terms, memory=memory, options=options)
    if named not in runs:
        raise LayoutError(f"the core does not run {lay} yet")
    return Choice(lay, costs or {str(lay): cost(lay)})


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
