"""The scheduler: admission, step plans and the requests' lives (ADR-0017; DESIGN.md, sections 2 and 4).

Each tick it admits what fits, builds one step plan of changes, submits it, and turns the core's events into
protocol lines. The rules, from ADR-0017:

- Prompt work is scheduled by time. While a request is writing, a prompt step aims at `prompt_budget_ms`, sized by
  the cost model; after it, decode is owed `decode_share` of its time before the next prompt step. A prompt chunk's
  time is mostly a fixed part that each piece pays, so a short prompt is read whole and a long one in pieces of at
  most `WHOLE_ROWS` (`CostModel.rows_within`). With nobody writing, a prompt takes whole chunks.
- Short prompts are packed into one step, shortest first, and the engine reads up to PROMPT_RUNS of them as one
  chunk. A prompt passed over `max_passes` times goes first.
- One producer per shared prefix: a request whose prompt starts with a prefix another request will checkpoint waits
  for that checkpoint, holding no slot and no pages, unless it has the higher priority.
- Checkpoints are taken where the model's rule puts them; a prompt span ends at each one.

The core keeps decoding between plans, so a plan is built only when something changes.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

import protocol
from engine.api.interface import (
    MAX_PAGE_CHANGES,
    MAX_PROMPT_SPANS,
    NO_SLOT,
    PAGE_CELLS,
    PROMPT_RUNS,
    WINDOW_ROWS,
    Engine,
    EngineError,
    EngineFacts,
    Event,
    EventKind,
    FinishReason,
    Image,
    PromptSpan,
    SlotSettings,
    StepPlan,
    prefix_key,
)
from engine.seqstate.manager import SeqState, pages_for
from engine.timing import STRETCH, CostModel

type CheckpointRule = Callable[[tuple[int, ...]], list[tuple[int, bool]]]
"""Where to checkpoint a prompt: (position, kept) pairs. The model's rule is in models/<arch>/."""

EVERY = 16384
"""A checkpoint every 16K prompt tokens (ADR-0017, item 6)."""


def every_16k(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
    return [(p, False) for p in range(EVERY, len(prompt), EVERY)]


class EngineFailed(RuntimeError):
    """The core reported an error of the engine as a whole."""


@dataclass(frozen=True, slots=True)
class SchedulerOptions:
    prompt_budget_ms: float = 500.0
    """What a prompt step aims at while others write (Splash's value, ADR-0017)."""
    decode_share: float = 0.5
    """Decode is owed this share of each prompt step's time (Splash's value)."""
    chunk_rows: int = 2048
    max_passes: int = 4
    decode_lookahead: int = 64
    """Cells assigned ahead of a decoding slot, so the core never waits for the host."""
    decode_batch: int = 256
    """Cells assigned at once when a slot's lookahead runs short. Each plan with page changes costs the core a write to
    the cards' page tables (ADR-0033), so they come in batches."""


class Phase(Enum):
    WAITING = auto()
    PREFILL = auto()
    DECODE = auto()


_FINISH = {
    FinishReason.EOS: protocol.Finish.STOP,
    FinishReason.LENGTH: protocol.Finish.LENGTH,
    FinishReason.CANCELLED: protocol.Finish.CANCEL,
}


@dataclass(slots=True)
class Request:
    rid: int
    prompt: tuple[int, ...]
    max_new: int
    settings: protocol.Settings
    arrived: float
    priority: int = 0
    phase: Phase = Phase.WAITING
    slot: int = -1
    next_pos: int = 0
    """The next prompt row to schedule."""
    done_pos: int = 0
    """Prompt rows now in the slot's state."""
    marks: list[tuple[int, bool]] = field(default_factory=lambda: [])
    """Checkpoint positions still ahead of next_pos, and whether each is kept."""
    passes: int = 0
    generated: int = 0
    reused: int = 0
    cancelled: bool = False
    stop_sent: bool = False
    waiting_for: int | None = None
    t_admit: float = 0.0
    t_first: float | None = None
    images: tuple[Image, ...] = ()
    key: tuple[int, ...] = ()
    """The prompt as prefixes are compared: with each image's tokens standing for its pixels (`prefix_key`)."""

    @property
    def remaining(self) -> int:
        return len(self.prompt) - self.next_pos

    def next_mark(self) -> int:
        return self.marks[0][0] if self.marks else len(self.prompt)


class Scheduler:
    def __init__(
        self,
        engine: Engine,
        facts: EngineFacts,
        options: SchedulerOptions | None = None,
        checkpoint_rule: CheckpointRule = every_16k,
        clock: Callable[[], float] = time.monotonic,
        *,
        cost: CostModel | None = None,
    ) -> None:
        self.engine = engine
        self.facts = facts
        self.options = options or SchedulerOptions()
        self.rule = checkpoint_rule
        self.clock = clock
        self.state = SeqState(facts.slots, facts.pages, facts.checkpoints, classes=max(1, facts.cards))
        self.cost = cost or CostModel.initial()
        self.waiting: list[Request] = []
        self.by_rid: dict[int, Request] = {}
        self.by_slot: dict[int, Request] = {}
        self.owed_ms = 0.0
        self.spans_in_flight: dict[int, int] = {}
        self.prompt_step_open = False
        """A plan with prompt spans is submitted and its STEP_DONE, which says what decode it owes, has not come."""
        self._serial = 0

    @property
    def idle(self) -> bool:
        return not self.by_rid

    def report(self) -> dict[str, str]:
        """The STATS line: the engine's counters, field for field, then the scheduler's: the requests waiting for a
        slot, reading their prompts and writing, and the cost model's parts in milliseconds."""
        out = {k: str(v) for k, v in dataclasses.asdict(self.engine.stats()).items()}
        phases = [r.phase for r in self.by_slot.values()]
        out["waiting"] = str(len(self.waiting))
        out["reading"] = str(phases.count(Phase.PREFILL))
        out["writing"] = str(phases.count(Phase.DECODE))
        out["decode_ms"] = f"{self.cost.decode_ms.value:.3f}"
        out["chunk_ms"] = f"{self.cost.chunk_ms:.3f}"
        out["row_ms"] = f"{self.cost.row_ms:.4f}"
        out["owed_ms"] = f"{self.owed_ms:.3f}"
        return out

    # Commands.

    def add(self, gen: protocol.Gen, priority: int = 0, images: tuple[Image, ...] = ()) -> list[protocol.Event]:
        if gen.rid in self.by_rid:
            return [protocol.Error(gen.rid, "request id already in use")]
        if gen.max_new < 1:
            return [protocol.Error(gen.rid, "max_new must be at least 1")]
        if len(gen.prompt) + gen.max_new > self.facts.max_context:
            return [
                protocol.Error(
                    gen.rid,
                    f"prompt ({len(gen.prompt)}) plus max_new ({gen.max_new}) exceeds the context "
                    f"({self.facts.max_context})",
                )
            ]
        if (problem := self._image_problem(gen.prompt, images)) is not None:
            return [protocol.Error(gen.rid, problem)]
        r = Request(gen.rid, gen.prompt, gen.max_new, gen.settings.with_defaults(), self.clock(), priority)
        r.images = images
        r.key = prefix_key(gen.prompt, images)
        self.by_rid[r.rid] = r
        self.waiting.append(r)
        return []

    def _image_problem(self, prompt: tuple[int, ...], images: tuple[Image, ...]) -> str | None:
        if not images:
            return None
        if not self.facts.image_tokens_max:
            return "this engine reads no images"
        end = 0
        for i, im in enumerate(images):
            try:
                im.check()
            except ValueError as exc:
                return f"image {i}: {exc}"
            if im.tokens > self.facts.image_tokens_max:
                return f"image {i} has {im.tokens} tokens, more than {self.facts.image_tokens_max}"
            if im.first < end or im.first + im.tokens > len(prompt):
                return f"image {i}: its tokens at {im.first} are out of order or past the prompt"
            end = im.first + im.tokens
        return None

    def cancel(self, rid: int) -> list[protocol.Event]:
        r = self.by_rid.get(rid)
        if r is None:
            return []
        if r.phase is Phase.WAITING:
            self.waiting.remove(r)
            del self.by_rid[rid]
            return [protocol.Done(rid, 0, len(r.prompt), 0.0, 0.0, protocol.Finish.CANCEL)]
        r.cancelled = True
        return []

    def refuse_waiting(self, message: str) -> list[protocol.Event]:
        """Every request not yet admitted ends with an error; the admitted ones go on."""
        out: list[protocol.Event] = []
        for r in list(self.waiting):
            self.waiting.remove(r)
            del self.by_rid[r.rid]
            out.append(protocol.Error(r.rid, message))
        return out

    # The tick.

    def tick(self, timeout_s: float) -> list[protocol.Event]:
        out: list[protocol.Event] = []
        self._admit(out)
        self._plan()
        for event in self.engine.wait(timeout_s):
            self._handle(event, out)
        return out

    def _producer_for(self, r: Request, have: int) -> int | None:
        """The request r should wait for: one that will checkpoint a prefix of r's prompt longer than `have`, or
        has one in progress. -1 for a checkpoint whose producer has moved on but which the core has not answered."""
        for p in self.by_rid.values():
            if p is r or p.phase is Phase.WAITING or p.priority < r.priority:
                continue
            for position, _ in p.marks:
                if have < position < len(r.key) and p.key[:position] == r.key[:position]:
                    return p.rid
        for ck in self.state.checkpoints:
            n = len(ck.tokens)
            if not ck.ready and have < n < len(r.key) and r.key[:n] == ck.tokens:
                return -1
        return None

    def _admit(self, out: list[protocol.Event]) -> None:
        self.waiting.sort(key=lambda r: (-r.priority, r.arrived))
        for r in list(self.waiting):
            best = self.state.checkpoints.longest_prefix(r.key)
            producer = self._producer_for(r, len(best.tokens) if best else 0)
            if producer is not None:
                r.waiting_for = producer
                continue
            r.waiting_for = None
            admission = self.state.plan_admission(r.key, r.max_new)
            if admission is None:
                break
            self._flush_drops()
            seq = self.state.admit(admission, r.key, r.max_new)
            s = r.settings
            try:
                self.engine.admit(
                    seq.slot,
                    SlotSettings(
                        s.temperature,
                        s.top_p,
                        s.top_k,
                        s.min_p,
                        s.penalty_last_n,
                        s.penalty_repeat,
                        s.penalty_freq,
                        s.penalty_present,
                        s.seed,
                        r.max_new,
                    ),
                    r.prompt,
                    r.images,
                )
            except EngineError as exc:
                self.state.release(seq.slot)
                self.waiting.remove(r)
                del self.by_rid[r.rid]
                out.append(protocol.Error(r.rid, str(exc)))
                continue
            self.waiting.remove(r)
            r.phase = Phase.PREFILL
            r.slot = seq.slot
            r.next_pos = r.done_pos = r.reused = seq.start
            r.t_admit = self.clock()
            r.marks = sorted((p, k) for p, k in self.rule(r.prompt) if seq.start < p < len(r.prompt))
            self.by_slot[seq.slot] = r
            if admission.checkpoint is not None:
                self.engine.restore(seq.slot, admission.checkpoint.id)
                out.append(protocol.Resume(r.rid, seq.start))

    def _flush_drops(self) -> None:
        for checkpoint_id in self.state.drops:
            self.engine.drop(checkpoint_id)
        self.state.drops.clear()

    def _plan(self) -> None:
        plan = StepPlan(self._serial + 1)
        for r in self.by_slot.values():
            if r.cancelled and not r.stop_sent:
                plan.stop.append(r.slot)
                r.stop_sent = True
            elif r.phase is Phase.DECODE:
                self._grow_ahead(r, len(r.prompt) + r.generated)
        marks = self._prompt_spans(plan)
        plan.pages = self.state.take_changes(MAX_PAGE_CHANGES)
        if plan.empty:
            return
        plan.kv_free = self.state.pool.size - self.state.pool.high
        self._serial = plan.serial
        self.engine.submit(plan)
        self.state.submitted()
        for span in plan.prompt:
            self.spans_in_flight[span.slot] = self.spans_in_flight.get(span.slot, 0) + 1
        self.prompt_step_open |= bool(plan.prompt)
        for r, position, kept in marks:
            ck = self.state.checkpoint(r.slot, position, kept)
            self._flush_drops()
            if ck is not None:
                self.engine.checkpoint(r.slot, ck.id)
        self._flush_drops()

    def _grow_ahead(self, r: Request, used: int) -> None:
        """Pages for the cells after the first `used`: at least `decode_lookahead`, then `decode_batch` more."""
        o = self.options
        if self.state.sequences[r.slot].cells < used + o.decode_lookahead:
            self.state.grow(r.slot, used + o.decode_lookahead + o.decode_batch)

    def _prompt_spans(self, plan: StepPlan) -> list[tuple[Request, int, bool]]:
        """Fill the plan's prompt spans. Returns the checkpoints to take after it."""
        if self.spans_in_flight:
            return []
        candidates = [
            r for r in self.by_slot.values() if r.phase is Phase.PREFILL and r.remaining > 0 and not r.cancelled
        ]
        if not candidates:
            return []
        writing = any(r.phase is Phase.DECODE for r in self.by_slot.values())
        if not writing:
            self.owed_ms = 0.0
        elif self.owed_ms > 0 or self.prompt_step_open:
            return []  # a span can be done before its step: the next prompt step waits for the decode it owes
        o = self.options
        candidates.sort(key=lambda r: (r.passes < o.max_passes, r.remaining, r.arrived))
        budget = o.chunk_rows
        left_ms = STRETCH * o.prompt_budget_ms
        marks: list[tuple[Request, int, bool]] = []
        chosen: set[int] = set()
        for r in candidates:
            if budget <= 0 or len(plan.prompt) >= MAX_PROMPT_SPANS:
                break
            rows = min(r.remaining, budget, r.next_mark() - r.next_pos)
            if writing:
                # While others write, the first prompt's piece is sized by time and is whole when it is short
                # (CostModel.rows_within); another prompt joins whole while the plan's prompt time stays in bound.
                # A longer piece shares the chunk of those before it in its PROMPT_RUNS, and so its fixed part.
                if not plan.prompt:
                    rows = self.cost.rows_within(o.prompt_budget_ms, rows)
                    ms = self.cost.prompt_ms(rows)
                else:
                    group = plan.prompt[len(plan.prompt) // PROMPT_RUNS * PROMPT_RUNS :]
                    shared = rows > WINDOW_ROWS and any(span.count > WINDOW_ROWS for span in group)
                    ms = self.cost.row_ms * rows if shared else self.cost.prompt_ms(rows)
                    if rows < r.next_mark() - r.next_pos or ms > left_ms:
                        break
                left_ms -= ms
            seq = self.state.sequences[r.slot]
            room = MAX_PAGE_CHANGES - self.state.pending_changes
            fits = (len(seq.pages) + room) * PAGE_CELLS - r.next_pos
            rows = min(rows, fits)
            if rows <= 0:
                break
            self.state.grow(r.slot, r.next_pos + rows)
            if pages_for(r.next_pos + rows) > len(seq.pages):
                raise AssertionError("prompt span without its pages")
            last = r.next_pos + rows == len(r.prompt)
            if last:
                self._grow_ahead(r, len(r.prompt))  # its first tokens decode without waiting for a plan
            plan.prompt.append(PromptSpan(r.slot, r.next_pos, rows, last))
            r.next_pos += rows
            budget -= rows
            chosen.add(r.rid)
            if r.marks and r.marks[0][0] == r.next_pos:
                position, kept = r.marks.pop(0)
                marks.append((r, position, kept))
        for r in candidates:
            r.passes = 0 if r.rid in chosen else r.passes + 1
        return marks

    # Events.

    def _handle(self, event: Event, out: list[protocol.Event]) -> None:
        match event.kind:
            case EventKind.STEP_DONE:
                ms = event.t_ns / 1e6
                self.cost.observe(event.a, event.b, ms, event.c)
                if event.b:
                    self.prompt_step_open = False
                writing = any(r.phase is Phase.DECODE for r in self.by_slot.values())
                if event.b and writing:
                    prompt_ms = max(0.0, ms - (self.cost.decode_ms.value if event.a else 0.0))
                    self.owed_ms += self.options.decode_share * prompt_ms
                elif not event.b:
                    self.owed_ms = max(0.0, self.owed_ms - ms)
                return
            case EventKind.CHECKPOINT_DONE:
                ck = self.state.checkpoints.get(event.a)
                if ck is not None:
                    ck.ready = True
                return
            case EventKind.ERROR if event.slot == NO_SLOT:
                raise EngineFailed(f"the core failed with error {event.a}")
            case _:
                pass
        r = self.by_slot.get(event.slot)
        if r is None:
            return
        now = self.clock()
        match event.kind:
            case EventKind.TOKEN:
                if r.t_first is None:
                    r.t_first = now
                r.phase = Phase.DECODE
                r.generated += 1
                out.append(protocol.Tokens(r.rid, (event.a,)))
            case EventKind.PROMPT_DONE:
                r.done_pos = event.a
                self._span_done(r.slot)
                out.append(protocol.Progress(r.rid, event.a, len(r.prompt), (now - r.t_admit) * 1e3))
            case EventKind.FINISHED:
                first = r.t_first if r.t_first is not None else now
                st = self.engine.slot_stats(r.slot)  # before the release empties the slot
                out.append(
                    protocol.Done(
                        r.rid,
                        r.generated,
                        len(r.prompt),
                        (first - r.t_admit) * 1e3,
                        (now - first) * 1e3,
                        _FINISH[FinishReason(event.a)],
                        r.reused,
                        (r.t_admit - r.arrived) * 1e3,
                        st.steps,
                        st.drafts,
                        st.accepted,
                        st.lookup_drafts,
                        st.lookup_accepted,
                    )
                )
                self._retire(r)
            case EventKind.STALLED:
                self._grow_ahead(r, max(event.a + 1, len(r.prompt) + r.generated))
            case EventKind.ERROR:
                out.append(protocol.Error(r.rid, f"the core failed this request with error {event.a}"))
                self._retire(r)

    def _span_done(self, slot: int) -> None:
        left = self.spans_in_flight.get(slot, 0) - 1
        if left > 0:
            self.spans_in_flight[slot] = left
        else:
            self.spans_in_flight.pop(slot, None)

    def _retire(self, r: Request) -> None:
        self.engine.release(r.slot)
        self.state.release(r.slot)
        self.spans_in_flight.pop(r.slot, None)
        del self.by_slot[r.slot]
        del self.by_rid[r.rid]
