"""The mock engine: the interface in Python, with no card (ADR-0022, item 6; Strata's MockEngine idea).

It runs the control plane's plans as the core would, and checks them:

- every prompt row and decode token is written into the cell its slot's page table names, and at the end of each
  prompt span and at each finish the slot's whole history is read back through the page table;
- a restored checkpoint must hold a prefix of the slot's prompt;
- prompt spans must follow each other without gaps.

A broken plan gives an ERROR event, as the core would. The "state" of a slot is a hash of every token it has
processed, so a sequence resumed from a checkpoint produces the same tokens as one computed from the start.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from engine.api.interface import (
    NO_SLOT,
    PAGE_CELLS,
    PROMPT_RUNS,
    TIMING_DTYPE,
    WINDOW_ROWS,
    EngineError,
    EngineFacts,
    EngineStats,
    ErrorCode,
    Event,
    EventKind,
    FinishReason,
    Image,
    OpenConfig,
    PageOp,
    SlotSettings,
    SlotStats,
    StepPlan,
    TimingRecords,
)

type Script = Callable[[Sequence[int]], int]
"""Given a slot's tokens so far (prompt and generated), the next token."""

_MASK = (1 << 64) - 1


def mix(h: int, x: int) -> int:
    """splitmix64 of h ^ x: a stand-in for the state a token leaves behind."""
    z = (h ^ x) + 0x9E3779B97F4A7C15 & _MASK
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & _MASK
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & _MASK
    return z ^ (z >> 31)


class PlanFault(Exception):
    """A plan the core would refuse."""

    def __init__(self, slot: int, message: str) -> None:
        super().__init__(message)
        self.slot = slot


@dataclass(slots=True)
class _Slot:
    settings: SlotSettings
    prompt: tuple[int, ...]
    tokens: list[int] = field(default_factory=lambda: [])
    """Every token processed into the state, in order: the positions 0..n-1."""
    state: int = 0
    table: dict[int, int] = field(default_factory=lambda: {})
    decoding: bool = False
    finished: bool = False
    stalled: bool = False
    pending: int | None = None
    """The last sampled token, not yet processed (it is processed at the next decode step)."""
    generated: int = 0
    steps: int = 0
    """Decode steps that processed one of its own tokens."""


@dataclass(slots=True)
class _Checkpoint:
    state: int
    tokens: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _Op:
    kind: str
    slot: int = 0
    checkpoint: int = 0


class MockEngine:
    def __init__(
        self,
        script: Script | None = None,
        *,
        prompt_row_ms: float = 0.46,
        decode_step_ms: float = 11.0,
        prompt_chunk_ms: float = 0.0,
        realtime: bool = False,
    ) -> None:
        """`prompt_row_ms` and `decode_step_ms` are the step times it reports (estimates from stand-in kernels:
        docs/measurements/MEASUREMENTS.md#2026-10-13-design-inputs), and `prompt_chunk_ms` the fixed time of a
        chunk, as the core's staging costs: the spans longer than the plan's window_rows of each PROMPT_RUNS of a plan
        share one. With `realtime`, it also sleeps for them."""
        self._script = script
        self._row_ms = prompt_row_ms
        self._chunk_ms = prompt_chunk_ms
        self._step_ms = decode_step_ms
        self._realtime = realtime
        self._config: OpenConfig | None = None
        self._slots: dict[int, _Slot] = {}
        self._pages: dict[int, list[int | None]] = {}
        self._checkpoints: dict[int, _Checkpoint] = {}
        self._queue: deque[StepPlan | _Op] = deque()
        self._events: deque[Event] = deque()
        self._timing: list[tuple[int, int, int, int]] = []
        self._stats = EngineStats()
        self._t0 = time.monotonic_ns()
        self._step = 0
        self._serial = -1
        self._asleep = False

    @property
    def config(self) -> OpenConfig:
        if self._config is None:
            raise EngineError(ErrorCode.WRONG_STATE, "not open")
        if self._asleep:
            raise EngineError(ErrorCode.WRONG_STATE, "asleep")
        return self._config

    def open(self, config: OpenConfig) -> EngineFacts:
        if self._config is not None:
            raise EngineError(ErrorCode.WRONG_STATE, "already open")
        self._config = config
        return self._facts(config)

    def sleep(self) -> None:
        _ = self.config
        if self._slots:
            raise EngineError(ErrorCode.WRONG_STATE, f"slots {sorted(self._slots)} are admitted")
        self._queue.clear()
        self._checkpoints.clear()
        self._pages.clear()
        self._asleep = True

    def wake(self) -> EngineFacts:
        if self._config is None or not self._asleep:
            raise EngineError(ErrorCode.WRONG_STATE, "not asleep")
        self._asleep = False
        self._serial = -1
        return self._facts(self._config)

    @staticmethod
    def _facts(config: OpenConfig) -> EngineFacts:
        return EngineFacts(
            cards=0,
            slots=config.slots,
            max_context=config.max_context,
            pages=config.pages,
            checkpoints=config.checkpoints,
            experts_cached=config.experts_cached,
            vram_used=(),
            vram_free=(),
            host_pinned=0,
            timing_fd=-1,
            version="mock",
        )

    def admit(self, slot: int, settings: SlotSettings, prompt: tuple[int, ...], images: tuple[Image, ...] = ()) -> None:
        config = self.config
        if not 0 <= slot < config.slots:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"slot {slot} out of range")
        if slot in self._slots:
            raise EngineError(ErrorCode.WRONG_STATE, f"slot {slot} is in use")
        if not prompt:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, "empty prompt")
        if len(prompt) + settings.max_new > config.max_context:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, "prompt and max_new exceed the context")
        if any(not 0 <= t < config.vocab_size for t in prompt):
            raise EngineError(ErrorCode.INVALID_ARGUMENT, "token id out of the vocabulary")
        self._slots[slot] = _Slot(settings, prompt)

    def release(self, slot: int) -> None:
        if self._slots.pop(slot, None) is None:
            raise EngineError(ErrorCode.WRONG_STATE, f"slot {slot} is not admitted")

    def submit(self, plan: StepPlan) -> None:
        plan.check()
        if plan.serial <= self._serial:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"plan serial {plan.serial} after {self._serial}")
        self._serial = plan.serial
        self._queue.append(plan)

    def checkpoint(self, slot: int, checkpoint_id: int) -> None:
        self._check_checkpoint_id(checkpoint_id)
        self._queue.append(_Op("checkpoint", slot, checkpoint_id))

    def restore(self, slot: int, checkpoint_id: int) -> None:
        self._check_checkpoint_id(checkpoint_id)
        self._queue.append(_Op("restore", slot, checkpoint_id))

    def drop(self, checkpoint_id: int) -> None:
        self._check_checkpoint_id(checkpoint_id)
        self._queue.append(_Op("drop", 0, checkpoint_id))

    def _check_checkpoint_id(self, checkpoint_id: int) -> None:
        if not 0 <= checkpoint_id < self.config.checkpoints:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"checkpoint {checkpoint_id} out of range")

    def poll(self, max_events: int = 256) -> list[Event]:
        if not self._events:
            self._advance()
        out: list[Event] = []
        while self._events and len(out) < max_events:
            out.append(self._events.popleft())
        return out

    def wait(self, timeout_s: float, max_events: int = 256) -> list[Event]:
        events = self.poll(max_events)
        if not events and timeout_s > 0:
            time.sleep(min(timeout_s, 0.001))
        return events

    def stats(self) -> EngineStats:
        """What the mock counts: plans, steps, rows and the times it reports; the rest stays zero."""
        return replace(self._stats, up_ns=time.monotonic_ns() - self._t0)

    def slot_stats(self, slot: int) -> SlotStats:
        s = self._slots.get(slot)
        return SlotStats() if s is None else SlotStats(steps=s.steps)

    def read_timing(self, max_records: int = 4096) -> TimingRecords:
        rows = self._timing[:max_records]
        del self._timing[:max_records]
        out = np.zeros(len(rows), dtype=TIMING_DTYPE)
        for i, (t, duration, step, aux) in enumerate(rows):
            out[i] = (t, duration, step, 0, 0, aux, 0)
        return out

    def close(self) -> None:
        self._config = None
        self._asleep = False
        self._slots.clear()
        self._queue.clear()

    @property
    def busy(self) -> bool:
        """Work is queued, or a slot is decoding."""
        return bool(self._queue) or any(s.decoding and not s.finished for s in self._slots.values())

    # The step.

    def _emit(self, kind: EventKind, slot: int, a: int = 0, b: int = 0, t_ns: int = 0, *, c: int = 0) -> None:
        self._events.append(Event(kind, slot, self._step, a, b, t_ns, c))

    def _advance(self) -> None:
        self._apply_ops()
        plan = self._queue.popleft() if self._queue and isinstance(self._queue[0], StepPlan) else None
        if plan is not None or any(s.decoding and not s.finished and not s.stalled for s in self._slots.values()):
            self._run_step(plan if isinstance(plan, StepPlan) else None)
        self._apply_ops()

    def _apply_ops(self) -> None:
        while self._queue and isinstance(self._queue[0], _Op):
            op = self._queue.popleft()
            assert isinstance(op, _Op)
            try:
                self._apply_op(op)
            except PlanFault as fault:
                self._emit(EventKind.ERROR, fault.slot, ErrorCode.INVALID_ARGUMENT)

    def _apply_op(self, op: _Op) -> None:
        match op.kind:
            case "checkpoint":
                s = self._slot(op.slot)
                self._checkpoints[op.checkpoint] = _Checkpoint(s.state, tuple(s.tokens))
                self._emit(EventKind.CHECKPOINT_DONE, op.slot, op.checkpoint)
            case "restore":
                s = self._slot(op.slot)
                ck = self._checkpoints.get(op.checkpoint)
                if ck is None:
                    raise PlanFault(op.slot, f"checkpoint {op.checkpoint} is empty")
                if s.tokens or s.decoding:
                    raise PlanFault(op.slot, "restore into a slot that has started")
                if s.prompt[: len(ck.tokens)] != ck.tokens or len(ck.tokens) >= len(s.prompt):
                    raise PlanFault(op.slot, f"checkpoint {op.checkpoint} is not a strict prefix of the prompt")
                s.state = ck.state
                s.tokens = list(ck.tokens)
            case "drop":
                self._checkpoints.pop(op.checkpoint, None)
            case _:
                raise AssertionError(op.kind)

    def _slot(self, slot: int) -> _Slot:
        s = self._slots.get(slot)
        if s is None:
            raise PlanFault(slot, f"slot {slot} is not admitted")
        return s

    def _apply_pages(self, plan: StepPlan) -> None:
        for change in plan.pages:
            s = self._slot(change.slot)
            match change.op:
                case PageOp.ASSIGN:
                    self._pages.setdefault(change.physical, [None] * PAGE_CELLS)
                    s.table[change.logical] = change.physical
                case PageOp.COPY:
                    source = self._pages.get(change.source)
                    if source is None:
                        raise PlanFault(change.slot, f"copy from page {change.source}, never written")
                    page = self._pages.setdefault(change.physical, [None] * PAGE_CELLS)
                    page[: change.cells] = source[: change.cells]
                    s.table[change.logical] = change.physical
                case PageOp.UNASSIGN:
                    for logical in [k for k in s.table if k >= change.logical]:
                        del s.table[logical]
            if s.stalled and change.op != PageOp.UNASSIGN:
                s.stalled = False

    def _write(self, slot: int, s: _Slot, token: int) -> bool:
        """Process one token at the slot's next position. False if its page is not assigned."""
        position = len(s.tokens)
        physical = s.table.get(position // PAGE_CELLS)
        if physical is None:
            return False
        self._pages[physical][position % PAGE_CELLS] = token
        s.tokens.append(token)
        s.state = mix(s.state, token)
        return True

    def _verify(self, slot: int, s: _Slot) -> None:
        """Read the slot's history back through its page table, as attention would."""
        for position, token in enumerate(s.tokens):
            physical = s.table.get(position // PAGE_CELLS)
            if physical is None:
                raise PlanFault(slot, f"position {position} has no page")
            if self._pages[physical][position % PAGE_CELLS] != token:
                raise PlanFault(slot, f"position {position} reads the wrong cell (page {physical})")

    def _sample(self, s: _Slot) -> int:
        if self._script is not None:
            return self._script([*s.tokens])
        h = s.state if s.settings.temperature == 0 or s.settings.top_k == 1 else mix(s.state, s.settings.seed)
        return h % self.config.vocab_size

    def _emit_token(self, slot: int, s: _Slot) -> None:
        try:
            token = self._sample(s)
        except Exception as exc:  # a script that does not know this sequence
            raise PlanFault(slot, f"script failed: {exc}") from None
        s.generated += 1
        s.pending = token
        self._emit(EventKind.TOKEN, slot, token, len(s.tokens))
        reason: FinishReason | None = None
        if token in self.config.eos_ids:
            reason = FinishReason.EOS
        elif s.generated >= s.settings.max_new:
            reason = FinishReason.LENGTH
        if reason is not None:
            self._finish(slot, s, reason)
        else:
            s.decoding = True

    def _finish(self, slot: int, s: _Slot, reason: FinishReason) -> None:
        self._verify(slot, s)
        s.finished = True
        s.decoding = False
        self._emit(EventKind.FINISHED, slot, reason)

    def _run_step(self, plan: StepPlan | None) -> None:
        self._step += 1
        rows = chunks = 0
        decoded = 0
        if plan is not None:
            try:
                self._apply_pages(plan)
                for slot in plan.stop:
                    s = self._slot(slot)
                    if not s.finished:
                        self._finish(slot, s, FinishReason.CANCELLED)
            except PlanFault as fault:
                self._emit(EventKind.ERROR, fault.slot, ErrorCode.INVALID_ARGUMENT)
            for span in plan.prompt:
                try:
                    rows += self._run_span(span.slot, span.first, span.count, span.last)
                except PlanFault as fault:
                    self._emit(EventKind.ERROR, fault.slot, ErrorCode.INVALID_ARGUMENT)
            window = plan.window_rows or WINDOW_ROWS
            for i in range(0, len(plan.prompt), PROMPT_RUNS):
                chunks += any(span.count > window for span in plan.prompt[i : i + PROMPT_RUNS])
        started = {span.slot for span in plan.prompt} if plan is not None else set[int]()
        for slot, s in self._slots.items():
            if not s.decoding or s.finished or slot in started:
                continue
            assert s.pending is not None
            if not self._write(slot, s, s.pending):
                if not s.stalled:
                    s.stalled = True
                    self._stats = replace(self._stats, stalls=self._stats.stalls + 1)
                    self._emit(EventKind.STALLED, slot, len(s.tokens))
                continue
            decoded += 1
            s.steps += 1
            try:
                self._emit_token(slot, s)
            except PlanFault as fault:
                self._emit(EventKind.ERROR, fault.slot, ErrorCode.INVALID_ARGUMENT)
        t_ms = (self._step_ms if decoded else 0.0) + rows * self._row_ms + chunks * self._chunk_ms
        if self._realtime:
            time.sleep(t_ms / 1e3)
        t_ns = int(t_ms * 1e6)
        st = self._stats
        self._stats = replace(
            st,
            plans=st.plans + 1,
            busy_ns=st.busy_ns + t_ns,
            prompt_chunks=st.prompt_chunks + chunks,
            prompt_rows=st.prompt_rows + rows,
            prompt_ns=st.prompt_ns + int(rows * self._row_ms * 1e6 + chunks * self._chunk_ms * 1e6),
            steps=st.steps + (1 if decoded else 0),
            step_rows=st.step_rows + decoded,
            step_ns=st.step_ns + (int(self._step_ms * 1e6) if decoded else 0),
        )
        self._timing.append((time.monotonic_ns(), t_ns, self._step, rows))
        self._emit(EventKind.STEP_DONE, NO_SLOT, decoded, rows, t_ns, c=chunks)

    def _run_span(self, slot: int, first: int, count: int, last: bool) -> int:
        s = self._slot(slot)
        if s.decoding or s.finished:
            raise PlanFault(slot, "prompt span for a slot past its prompt")
        if first != len(s.tokens):
            raise PlanFault(slot, f"span starts at {first}, the slot is at {len(s.tokens)}")
        if count <= 0 or first + count > len(s.prompt):
            raise PlanFault(slot, f"span {first}+{count} outside the prompt of {len(s.prompt)}")
        if last != (first + count == len(s.prompt)):
            raise PlanFault(slot, "the last flag does not match the prompt's end")
        for position in range(first, first + count):
            if not self._write(slot, s, s.prompt[position]):
                raise PlanFault(slot, f"prompt position {position} has no page")
        self._verify(slot, s)
        self._emit(EventKind.PROMPT_DONE, slot, len(s.tokens), count)
        if last:
            self._emit_token(slot, s)
        return count
