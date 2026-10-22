"""The scheduler against the mock engine, which checks every plan's pages and checkpoints as it runs them."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pytest

import protocol
from engine.api.interface import Event, EventKind, OpenConfig, StepPlan
from engine.api.mock import MockEngine, Script
from engine.scheduler import CheckpointRule, Phase, Scheduler, SchedulerOptions, every_16k
from engine.timing import WHOLE_ROWS
from tests.helpers import facts_of, open_config


class SpyEngine(MockEngine):
    def __init__(self, script: Script | None = None, *, prompt_chunk_ms: float = 0.0) -> None:
        super().__init__(script, prompt_chunk_ms=prompt_chunk_ms)
        self.plans: list[StepPlan] = []
        self.calls: list[tuple[str, int, int]] = []
        self.submitted_at: list[int] = []
        """The mock's step count when each plan was submitted."""

    def submit(self, plan: StepPlan) -> None:
        self.plans.append(plan)
        self.submitted_at.append(self._step)  # pyright: ignore[reportPrivateUsage]
        super().submit(plan)

    def checkpoint(self, slot: int, checkpoint_id: int) -> None:
        self.calls.append(("checkpoint", slot, checkpoint_id))
        super().checkpoint(slot, checkpoint_id)

    def restore(self, slot: int, checkpoint_id: int) -> None:
        self.calls.append(("restore", slot, checkpoint_id))
        super().restore(slot, checkpoint_id)


@dataclass
class Run:
    engine: SpyEngine
    scheduler: Scheduler
    events: list[protocol.Event] = field(default_factory=lambda: [])

    def add(self, rid: int, prompt: tuple[int, ...], max_new: int, **settings: float) -> None:
        self.events += self.scheduler.add(protocol.Gen(rid, max_new, prompt, protocol.Settings(**settings)))  # pyright: ignore[reportArgumentType]

    def tick(self, n: int = 1) -> None:
        for _ in range(n):
            self.events += self.scheduler.tick(0)

    def until_idle(self, limit: int = 100_000) -> None:
        for _ in range(limit):
            if self.scheduler.idle:
                return
            self.tick()
        raise AssertionError("the scheduler did not finish")

    def tokens(self, rid: int) -> list[int]:
        return [t for e in self.events if isinstance(e, protocol.Tokens) and e.rid == rid for t in e.ids]

    def done(self, rid: int) -> protocol.Done:
        (d,) = [e for e in self.events if isinstance(e, protocol.Done) and e.rid == rid]
        return d

    def errors(self) -> list[protocol.Error]:
        return [e for e in self.events if isinstance(e, protocol.Error)]


def make(
    config: OpenConfig | None = None,
    rule: CheckpointRule = every_16k,
    options: SchedulerOptions | None = None,
    script: Script | None = None,
    prompt_chunk_ms: float = 0.0,
) -> Run:
    config = config or open_config()
    engine = SpyEngine(script, prompt_chunk_ms=prompt_chunk_ms)
    facts = engine.open(config)
    assert facts_of(config).slots == facts.slots
    return Run(engine, Scheduler(engine, facts, options, rule))


def prompt(n: int, start: int = 3) -> tuple[int, ...]:
    return tuple(start + i % 900 for i in range(n))


def test_one_request():
    run = make()
    run.add(1, prompt(9), 5)
    run.until_idle()
    assert run.errors() == []
    assert len(run.tokens(1)) == 5
    d = run.done(1)
    assert (d.generated, d.prompt, d.finish) == (5, 9, protocol.Finish.LENGTH)
    assert run.scheduler.state.pool.free == run.scheduler.state.pool.size


def test_eos_finishes_with_stop():
    run = make(script=lambda tokens: 2 if len(tokens) >= 12 else 7)
    run.add(1, prompt(10), 50)
    run.until_idle()
    assert run.tokens(1) == [7, 7, 2]
    assert run.done(1).finish is protocol.Finish.STOP


def test_more_requests_than_slots_all_finish():
    run = make(open_config(slots=2))
    for rid in range(6):
        run.add(rid, prompt(5 + rid, start=rid * 10 + 3), 4)
    run.tick()
    assert sum(r.phase is not Phase.WAITING for r in run.scheduler.by_rid.values()) == 2
    run.until_idle()
    assert run.errors() == []
    assert all(len(run.tokens(rid)) == 4 for rid in range(6))
    assert run.scheduler.state.pool.free == run.scheduler.state.pool.size


def test_tokens_do_not_depend_on_company():
    """A request's tokens are the same alone and beside others (TESTING.md, level 6, as far as the mock goes)."""
    alone = make()
    alone.add(1, prompt(30), 8, temperature=0.0)
    alone.until_idle()
    crowd = make(open_config(slots=4))
    crowd.add(5, prompt(12, start=500), 8)
    crowd.add(1, prompt(30), 8, temperature=0.0)
    crowd.add(6, prompt(7, start=700), 8)
    crowd.until_idle()
    assert crowd.tokens(1) == alone.tokens(1)


def test_refuses_what_exceeds_the_context():
    run = make(open_config(context=64))
    run.add(1, prompt(60), 5)
    assert [type(e) for e in run.events] == [protocol.Error]
    assert run.scheduler.idle


def test_cancel_waiting_and_running():
    run = make(open_config(slots=1))
    run.add(1, prompt(8), 1000)
    run.add(2, prompt(8, start=50), 1000)
    run.tick(3)
    run.events += run.scheduler.cancel(2)
    assert run.done(2).finish is protocol.Finish.CANCEL
    run.events += run.scheduler.cancel(1)
    run.until_idle()
    assert run.done(1).finish is protocol.Finish.CANCEL
    assert 0 < len(run.tokens(1)) < 1000
    assert run.errors() == []


def test_short_prompts_are_packed():
    run = make(open_config(slots=4))
    for rid, n in enumerate((7, 5, 6)):
        run.add(rid, prompt(n, start=rid * 100 + 3), 2)
    run.tick()
    first = run.engine.plans[0]
    assert sorted(span.count for span in first.prompt) == [5, 6, 7]
    assert all(span.last for span in first.prompt)
    run.until_idle()
    assert run.errors() == []


def test_prompt_steps_are_sized_by_time_while_others_write():
    options = SchedulerOptions(prompt_budget_ms=100.0, decode_share=0.5)
    run = make(open_config(slots=2, context=8192), options=options)
    run.add(1, prompt(4), 400)
    run.tick(3)
    assert run.scheduler.by_rid[1].phase is Phase.DECODE
    run.add(2, prompt(3000, start=7), 2)
    run.until_idle()
    assert run.errors() == []
    steps = [(at, p) for at, p in zip(run.engine.submitted_at, run.engine.plans, strict=True) if p.prompt][1:]
    spans = [s.count for _, p in steps for s in p.prompt]
    assert sum(spans) == 3000
    assert max(spans) <= 100.0 / 0.46 + 1, "the mock reports 0.46 ms per row"
    gaps = [b - a for (a, _), (b, _) in itertools.pairwise(steps)]
    assert min(gaps) >= 4, f"decode is owed half of each ~100 ms prompt step, ~5 steps of 11 ms: {gaps}"


class LateStepDone(SpyEngine):
    """Each STEP_DONE comes one wait later than the events of its step, as the core's step event can come after its
    prompt events."""

    def __init__(self) -> None:
        super().__init__()
        self.held: list[Event] = []

    def wait(self, timeout_s: float, max_events: int = 256) -> list[Event]:
        events, self.held = self.held, []
        for e in super().wait(timeout_s, max_events):
            (self.held if e.kind is EventKind.STEP_DONE else events).append(e)
        return events


def test_a_prompt_step_waits_for_the_decode_the_step_before_owes():
    options = SchedulerOptions(prompt_budget_ms=100.0, decode_share=0.5)
    engine = LateStepDone()
    engine.open(open_config(slots=2, context=8192))
    run = Run(engine, Scheduler(engine, facts_of(open_config(slots=2, context=8192)), options, every_16k))
    run.add(1, prompt(4), 400)
    run.tick(3)
    run.add(2, prompt(3000, start=7), 2)
    run.until_idle()
    assert run.errors() == []
    steps = [at for at, p in zip(engine.submitted_at, engine.plans, strict=True) if p.prompt][1:]
    gaps = [b - a for a, b in itertools.pairwise(steps)]
    assert min(gaps) >= 4, f"no prompt step before the one before it is done and its decode paid: {gaps}"


def test_a_short_prompt_beside_a_writer_is_one_piece():
    options = SchedulerOptions(prompt_budget_ms=500.0)
    run = make(open_config(slots=3, context=8192), options=options, prompt_chunk_ms=600.0)
    run.add(1, prompt(300), 400)
    run.tick(3)
    assert run.scheduler.by_rid[1].phase is Phase.DECODE
    assert run.scheduler.cost.chunk_ms == pytest.approx(600.0), "the writer's prompt taught the model the fixed part"
    run.add(2, prompt(568, start=7), 2)
    run.add(3, prompt(5000, start=11), 2)
    run.until_idle()
    assert run.errors() == []
    pieces: dict[int, list[int]] = {}
    for p in run.engine.plans[1:]:
        for span in p.prompt:
            pieces.setdefault(span.slot, []).append(span.count)
    short, long = sorted(pieces.values(), key=len)
    assert short == [568], "its chunk is over the budget, but cutting it would pay the fixed part twice"
    assert sum(long) == 5000
    assert max(long) <= WHOLE_ROWS, "a long prompt holds the writers for one piece of at most WHOLE_ROWS at a time"
    assert min(long[:-1]) >= WHOLE_ROWS // 2, "no piece so short that it is almost all fixed part"


def test_short_prompts_beside_a_writer_share_a_chunk():
    options = SchedulerOptions(prompt_budget_ms=500.0)
    run = make(open_config(slots=4, context=8192), options=options, prompt_chunk_ms=600.0)
    run.add(1, prompt(300), 400)
    run.tick(3)
    assert run.scheduler.by_rid[1].phase is Phase.DECODE
    run.add(2, prompt(100, start=7), 2)
    run.add(3, prompt(120, start=11), 2)
    run.until_idle()
    assert run.errors() == []
    steps = [sorted(span.count for span in p.prompt) for p in run.engine.plans[1:] if p.prompt]
    assert steps == [[100, 120]], "the second adds no fixed part to the step"


def test_a_long_prompt_alone_takes_whole_chunks():
    run = make(open_config(slots=1, context=8192))
    run.add(1, prompt(5000), 2)
    run.until_idle()
    spans = [s.count for p in run.engine.plans for s in p.prompt]
    assert spans == [2048, 2048, 904]


def marks_at(*positions: int) -> CheckpointRule:
    return lambda p: [(x, False) for x in positions if x < len(p)]


def test_resume_from_a_checkpoint_gives_the_same_tokens():
    shared = prompt(10)
    tail = (777, 778, 779)
    fresh = make()
    fresh.add(2, shared + tail, 6, temperature=0.0)
    fresh.until_idle()

    run = make(rule=marks_at(10))
    run.add(1, (*shared, 5, 6), 3)
    run.until_idle()
    run.add(2, shared + tail, 6, temperature=0.0)
    run.until_idle()
    assert run.errors() == []
    assert protocol.Resume(2, 10) in run.events
    assert run.done(2).reused == 10
    assert run.tokens(2) == fresh.tokens(2)
    assert [c[0] for c in run.engine.calls] == ["checkpoint", "restore"]


def test_one_producer_per_shared_prefix():
    shared = prompt(40)
    run = make(open_config(slots=2), rule=marks_at(40))
    run.add(1, (*shared, 1, 2), 3)
    run.add(2, (*shared, 8, 9), 3)
    run.tick()
    assert run.scheduler.by_rid[2].phase is Phase.WAITING
    assert run.scheduler.by_rid[2].waiting_for == 1
    run.until_idle()
    assert run.errors() == []
    assert run.done(2).reused == 40
    assert run.done(1).reused == 0


def test_prompt_spans_end_at_checkpoints():
    run = make(open_config(context=8192), rule=marks_at(100, 3000))
    run.add(1, prompt(5000), 1)
    run.until_idle()
    spans = [(s.first, s.count) for p in run.engine.plans for s in p.prompt]
    assert spans == [(0, 100), (100, 2048), (2148, 852), (3000, 2000)]
    assert [c[0] for c in run.engine.calls] == ["checkpoint", "checkpoint"]


def test_checkpoints_are_evicted_without_breaking_requests():
    run = make(open_config(slots=2, checkpoints=2), rule=marks_at(8, 16))
    for rid in range(8):
        run.add(rid, prompt(24, start=rid * 50 + 3), 2)
    run.until_idle()
    assert run.errors() == []
    assert len(run.scheduler.state.checkpoints) <= 2


def test_admission_waits_for_pages():
    run = make(open_config(slots=2, context=64, pages=12))
    run.add(1, prompt(20), 20)
    run.add(2, prompt(20, start=100), 20)
    run.tick()
    assert run.scheduler.by_rid[2].phase is Phase.WAITING, "10 pages each, 12 in all"
    run.until_idle()
    assert run.errors() == []
    assert len(run.tokens(2)) == 20


@pytest.mark.parametrize("lookahead", [1, 4, 16])
def test_decode_never_runs_out_of_pages(lookahead: int):
    run = make(options=SchedulerOptions(decode_lookahead=lookahead), script=lambda _: 7)
    run.add(1, prompt(3), 200)
    run.until_idle()
    assert run.errors() == []
    assert len(run.tokens(1)) == 200
