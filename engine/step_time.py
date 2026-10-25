"""The step-time model: how long one decode step takes on a layout of the cards, for a load of rows, before it runs.

A layout puts the cards in groups. Each group holds a share of the layers, and each card of a group a slice of each
of those layers. The model costs a layout from the model's products, the memory plan and measured terms; the layout
chooser (layout.py) calls it at load. The model supplies its products as a `ModelSteps` and its terms as a `Terms`
(models/<arch>/steps.py).
"""

from __future__ import annotations

import contextlib
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from engine.memory_plan import ModelMemory, PlanError, PlanOptions, plan


class LayoutError(ValueError):
    """The layout breaks a constraint."""


class Class(StrEnum):
    """The kinds of weight read a step's time is summed over."""

    GATE_UP = "routed gate and up"
    DOWN = "routed down"
    DENSE = "dense"
    HEAD = "head"


@dataclass(frozen=True, slots=True)
class Product:
    """One weight product the cards run in a step."""

    kernel: str
    """The key of its bandwidth in `Terms.rates`."""
    kind: Class
    nbytes: int
    """The bytes it reads for one row, whole: for routed experts, one row's experts."""
    per_row: bool
    """Its bytes grow with the rows, as routed experts' do when each row has its own."""
    columns: int
    """For a product split by columns, the values in a row that the cards divide; 0 for one split by rows."""
    block: int
    """Values per quant block. A column split keeps whole blocks on each card."""


@dataclass(frozen=True, slots=True)
class ModelSteps:
    """A model's products for the step-time model."""

    layers: tuple[tuple[Product, ...], ...]
    head: tuple[Product, ...]
    """Read once a step, by the last group."""
    exchanges: int
    """Reduces a token makes over all layers."""
    experts: int
    """Routed experts a layer."""
    experts_a_token: int
    """Routed experts a row uses in each layer."""
    kv_heads: int


Rates = Mapping[int, Sequence[float | None]]
"""Rows -> GB/s at each of `Terms.widths`; None where not measured."""


@dataclass(frozen=True, slots=True)
class Terms:
    """The measured numbers the model is filled with. Times in µs unless the name says ms."""

    widths: tuple[int, ...]
    """The cards a group may have; `rates` gives one value for each, in this order."""
    rates: Mapping[str, Rates]
    """Each kernel's GB/s by rows and width."""
    reference: str
    """The kernel whose change with rows stands in for kernels measured at one row only."""
    exchange_us: Mapping[int, float]
    """Rows -> one peer write of a reduce."""
    handoff_us: float
    """One hand-off from a group to the next."""
    hit_rate: tuple[tuple[int, float], ...]
    """(Experts cached a layer, share of routed experts the cards serve), rising."""
    miss_layer_us: float
    """The CPU's cost for a layer with any miss, beside the cost per missed expert."""
    miss_expert_us: float
    cpu_wait_ms: Mapping[int, float]
    """Rows -> the measured wait for the CPU's misses a step, with `cpu_wait_experts` cached."""
    cpu_wait_experts: int
    step_ms: float
    """The measured one-row step: two cards in one group, `cpu_wait_experts` cached."""
    fixed_per_row_ms: float
    """What F grows by for each row past the first."""
    fixed_ms: float | None = None
    """F(1), the residual of `step_ms`; set by `calibrate`."""


@dataclass(frozen=True, slots=True)
class Layout:
    """`groups` groups of `width` cards. Each group holds layers / groups layers; each card 1 / width of each."""

    groups: int
    width: int
    experts_cached: int
    """Experts a layer each group caches, at most the layer's experts."""
    attention_by_heads: bool
    """Attention's products split by KV heads, possible while width <= KV heads; otherwise by rows."""

    def __str__(self) -> str:
        return f"groups({self.groups}, {self.width})"


@dataclass(frozen=True, slots=True)
class StepTime:
    """One step's predicted time, by term, in ms."""

    layout: Layout
    rows: int
    requests: int
    experts_cached: int
    fixed_ms: float
    bytes: Mapping[Class, int]
    """What the step reads in each class, over all cards."""
    transfer_ms: Mapping[Class, float]
    exchange_ms: float
    handoff_ms: float
    miss_ms: float
    pipelined: bool = False
    """Whether the groups pipeline rows from several requests; the core runs them in turn."""

    @property
    def step_ms(self) -> float:
        return self.fixed_ms + sum(self.transfer_ms.values()) + self.exchange_ms + self.handoff_ms + self.miss_ms

    @property
    def token_ms(self) -> float:
        """The step's time a row; pipelined groups take rows from as many requests as groups at once."""
        if not self.pipelined or self.requests < self.layout.groups:
            return self.step_ms / self.rows
        return self.step_ms / (self.rows * min(self.requests, self.layout.groups))

    def lines(self) -> list[str]:
        rows = f"{self.rows} row{'s' if self.rows > 1 else ''}"
        out = [f"{self.layout}, {rows} from {self.requests}, {self.experts_cached} experts cached a layer"]
        out.append(f"  F                {self.fixed_ms:7.3f} ms")
        for c in Class:
            out.append(f"  {c.value:<16} {self.transfer_ms[c]:7.3f} ms  {self.bytes[c] / 1e9:.3f} GB")
        out.append(f"  exchanges        {self.exchange_ms:7.3f} ms")
        out.append(f"  hand-offs        {self.handoff_ms:7.3f} ms")
        out.append(f"  CPU miss wait    {self.miss_ms:7.3f} ms")
        out.append(f"  step             {self.step_ms:7.3f} ms, {self.token_ms:.3f} ms a token")
        return out


def _linear(points: Sequence[tuple[float, float]], x: float) -> float:
    """Linear between rising points, and the end value outside them."""
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in itertools.pairwise(points):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def rate(terms: Terms, kernel: str, width: int, rows: int) -> float | None:
    """A kernel's GB/s at the width and rows, by the rows rule (`step_time`); None if not measured at the width."""
    i = terms.widths.index(width)
    own, ref = terms.rates[kernel], terms.rates[terms.reference]
    one, ref_one = own[1][i], ref[1][i]
    if one is None or ref_one is None:
        return None
    points: list[tuple[float, float]] = []
    for r in sorted(ref):
        v = own[r][i] if r in own else None
        if v is None:
            ref_r = ref[r][i]
            if ref_r is None:
                return None
            v = one * ref_r / ref_one
        points.append((r, v))
    return _linear(points, rows)


def hit_rate(terms: Terms, experts_cached: int) -> float:
    """The share of routed experts the cards serve: linear between the curve's points, from none at 0 cached."""
    return _linear([(0, 0.0), *terms.hit_rate], experts_cached)


def _miss_cost_ms(terms: Terms, model: ModelSteps, experts_cached: int, rows: int) -> float:
    h = hit_rate(terms, experts_cached)
    layers = len(model.layers)
    slots = model.experts_a_token * rows
    return _count_cost_ms(terms, layers * (1 - h**slots), layers * slots * (1 - h))


def _count_cost_ms(terms: Terms, layers_missed: float, misses: float) -> float:
    return (layers_missed * terms.miss_layer_us + misses * terms.miss_expert_us) / 1000


def _overlap(terms: Terms, model: ModelSteps, rows: int) -> float:
    """The overlap rule's share of the CPU's cost that the cards wait for, at `rows`."""
    points = [
        (r, wait / _miss_cost_ms(terms, model, terms.cpu_wait_experts, r))
        for r, wait in sorted(terms.cpu_wait_ms.items())
    ]
    return _linear(points, rows)


def miss_wait_ms(terms: Terms, model: ModelSteps, experts_cached: int, rows: int) -> float:
    """M: the CPU's cost for the misses, scaled by the overlap rule (`step_time`)."""
    return _miss_cost_ms(terms, model, experts_cached, rows) * _overlap(terms, model, rows)


def counted_miss_wait_ms(terms: Terms, model: ModelSteps, rows: int, layers_missed: float, misses: float) -> float:
    """M for misses counted in steps that ran, in place of the hit-rate curve's, scaled by the overlap rule."""
    return _count_cost_ms(terms, layers_missed, misses) * _overlap(terms, model, rows)


def _check_columns(where: str, products: Sequence[Product], width: int) -> None:
    for p in products:
        if p.columns and (p.columns % width or (p.columns // width) % p.block):
            raise LayoutError(
                f"{where}: {p.kernel} has {p.columns} columns, which {width} cards cannot share in whole blocks of "
                f"{p.block}"
            )


def layout(
    groups: int, width: int, cards: int, model: ModelSteps, terms: Terms, *, memory: ModelMemory, options: PlanOptions
) -> Layout:
    """`groups(groups, width)` on `cards` cards, with its experts cached a layer. Raises LayoutError when the layout
    breaks a constraint: the cards it uses, a width with no measured rate for some product, a layer count the groups
    do not divide, a column split that cuts a quant block, or a memory plan that does not fit."""
    if groups < 1 or width < 1 or groups * width != cards:
        raise LayoutError(f"groups({groups}, {width}) does not use {cards} cards")
    if width not in terms.widths:
        raise LayoutError(f"no rates at {width} cards a group")
    if len(model.layers) % groups:
        raise LayoutError(f"{groups} groups do not divide {len(model.layers)} layers")
    for n, products in enumerate(model.layers):
        _check_columns(f"layer {n}", products, width)
    _check_columns("the head", model.head, width)
    for p in {p for layer in model.layers for p in layer} | set(model.head):
        if p.kernel not in terms.rates:
            raise LayoutError(f"no rate measured for {p.kernel}")
        if rate(terms, p.kernel, width, 1) is None:
            raise LayoutError(f"{p.kernel} has no rate at {width} cards a group")
    try:
        mp = plan(memory, options, cards=width)
    except PlanError as e:
        raise LayoutError(str(e)) from e
    # A card holds 1 / width of each cached expert of its group's layers.
    cached = min(model.experts, mp.experts_cached * groups // len(model.layers))
    return Layout(groups, width, cached, attention_by_heads=width <= model.kv_heads)


def layouts(cards: int, model: ModelSteps, terms: Terms, *, memory: ModelMemory, options: PlanOptions) -> list[Layout]:
    """Every layout of `cards` cards that meets the constraints, fewest groups first."""
    found: list[Layout] = []
    for g in range(1, cards + 1):
        if cards % g == 0:
            with contextlib.suppress(LayoutError):
                found.append(layout(g, cards // g, cards, model, terms, memory=memory, options=options))
    return found


def _step(
    terms: Terms, model: ModelSteps, lay: Layout, rows: int, *, requests: int, experts_cached: int, fixed_ms: float
) -> StepTime:
    if not 1 <= requests <= rows:
        raise ValueError(f"{rows} rows from {requests} requests")
    w = lay.width
    nbytes = dict.fromkeys(Class, 0)
    ms = dict.fromkeys(Class, 0.0)
    for p in [p for layer in model.layers for p in layer] + list(model.head):
        b = p.nbytes * (rows if p.per_row else 1)
        gbs = rate(terms, p.kernel, w, rows)
        if gbs is None:
            raise LayoutError(f"{p.kernel} has no rate at {w} cards a group")
        nbytes[p.kind] += b
        ms[p.kind] += b / w / (gbs * 1e9) * 1e3
    exchange_us = _linear(sorted(terms.exchange_us.items()), rows)
    return StepTime(
        layout=lay,
        rows=rows,
        requests=requests,
        experts_cached=experts_cached,
        fixed_ms=fixed_ms,
        bytes=nbytes,
        transfer_ms=ms,
        exchange_ms=model.exchanges * (w - 1) * exchange_us / 1000,
        handoff_ms=(lay.groups - 1) * terms.handoff_us / 1000,
        miss_ms=miss_wait_ms(terms, model, experts_cached, rows),
    )


def calibrate(terms: Terms, model: ModelSteps) -> Terms:
    """The terms with F(1) set: the measured one-row step less the model's other terms for two cards in one group,
    one row from one request, `terms.cpu_wait_experts` cached. A calibration, not a measurement."""
    today = Layout(1, 2, terms.cpu_wait_experts, attention_by_heads=True)
    rest = _step(terms, model, today, 1, requests=1, experts_cached=terms.cpu_wait_experts, fixed_ms=0.0).step_ms
    return replace(terms, fixed_ms=terms.step_ms - rest)


def step_time(
    terms: Terms,
    model: ModelSteps,
    lay: Layout,
    rows: int,
    *,
    requests: int = 1,
    experts_cached: int | None = None,
    pipelined: bool = False,
) -> StepTime:
    """The predicted time of one step of `rows` rows from `requests` requests on the layout, and its time a token.

    `terms` must be calibrated (`calibrate`). `experts_cached` replaces the layout's own count, to hold M at another
    layout's value. `pipelined`: the groups pipeline rows from several requests, which the core does not do yet.
    Raises ValueError for requests outside 1..rows or uncalibrated terms.

        T_step = F(A) + sum over products of (bytes / width) / rate(width, A)
                 + exchanges * (width - 1) * t_x(A) + (groups - 1) * t_h + M(experts cached, A)

    A is the rows. Each group runs its layers in turn, so the products and the reduces of all layers add up; each
    reduce is width - 1 peer writes, none at width 1. M is the CPU's cost for the misses: a cost per layer with a miss
    and per missed expert, with layers missed = layers * (1 - h^(experts a token * A)) and misses = layers * experts a
    token * A * (1 - h), h from the hit-rate curve at the experts cached.

    Assumptions of the model, each to be checked by measurement:

    - Pipelining rule: with `pipelined`, rows from fewer requests than groups run the groups in turn, so a token takes
      T_step / A; rows from at least as many requests as groups pipeline across the groups, so a token takes
      T_step / (A * groups). The core runs the groups in turn whatever the requests: a token takes T_step / A.
    - Overlap rule: the CPU's work overlaps the cards', so the wait is less than its cost. M is the cost scaled by the
      measured wait over the cost at `cpu_wait_experts` cached and the same rows, linear between the measured rows.
    - F(A) = F(1) + (A - 1) * `fixed_per_row_ms`: everything else the cards do (router, norms, DeltaNet recurrence,
      launch gaps) and the host's work. F(1) is a residual (`calibrate`); both are calibration.
    - Rows rule: a rate is linear between the measured row counts and held at the last past them; a kernel without a
      value at some row count takes its one-row value times the reference kernel's change at that count and width.
      t_x is linear between its rows and held past them likewise. A width without a rate has no value: `layout`
      rejects it.
    - Nearest rate: a product of a type or shape with no rate of its own takes the nearest measured kernel's, as
      the model's `ModelSteps` names it.
    - Where attention splits by rows, past its KV heads, the split costs nothing of its own.
    """
    if terms.fixed_ms is None:
        raise ValueError("the terms are not calibrated")
    cached = lay.experts_cached if experts_cached is None else experts_cached
    fixed = terms.fixed_ms + (rows - 1) * terms.fixed_per_row_ms
    s = _step(terms, model, lay, rows, requests=requests, experts_cached=cached, fixed_ms=fixed)
    return replace(s, pipelined=pipelined)
