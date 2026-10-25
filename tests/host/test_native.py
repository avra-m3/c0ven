"""The native engine's adapter, driven by the scheduler, over a fake of the core that follows core/api/engine_api.cpp:
a plan is one step, its page changes first, then its prompt pieces and one decode step for the slots it names; a slot
is sampled after the last row of its prompt and after each decode row; checkpoint and restore wait for the queued
plans. The fake keeps each position's token in its page, and samples from what it reads back through the slot's page
table, so a wrong table gives wrong tokens; a slot past its pages stalls."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import pytest

import protocol
from engine.api import abi
from engine.api.interface import EngineError, ErrorCode, EventKind, Image, OpenConfig, SlotSettings
from engine.api.native import NativeEngine
from engine.scheduler import CheckpointRule, Scheduler, SchedulerOptions, every_16k
from tests.helpers import open_config

VOCAB = 1000
EOS = 2
GREEDY = SlotSettings(0.0, 1.0, 1, 0.0, 0, 1.0, 0.0, 0.0, 0, 4)


def next_token(tokens: Sequence[int]) -> int:
    """The fake model: a function of every token so far, never EOS on its own."""
    h = 17
    for t in tokens:
        h = (h * 31 + t) % 1_000_003
    return 3 + h % (VOCAB - 3)


@dataclass
class _Slot:
    settings: abi.CSettings
    prompt: list[int]
    tokens: list[int] = field(default_factory=lambda: [])
    next: int = -1
    generated: int = 0
    done: bool = False
    stalled: bool = False
    steps: int = 0
    """Decode steps that fed it one of its own tokens."""


@dataclass
class FakeCore:
    prompt_chunk: int = 64
    max_ctx: int = 4096
    slots: dict[int, _Slot] = field(default_factory=lambda: {})
    plans: deque[tuple[abi.CStepPlan, list[abi.CPageChange]]] = field(default_factory=deque)
    events: deque[abi.CEvent] = field(default_factory=deque)
    timing: deque[abi.CTiming] = field(default_factory=deque)
    checkpoints: dict[int, list[int]] = field(default_factory=lambda: {})
    seq: int = 0
    decode_steps: int = 0
    decode_rows: int = 0
    log: list[tuple[list[tuple[int, int]], list[int]]] = field(default_factory=lambda: [])
    """Each plan run: its pieces and its decode slots."""
    queue_limit: int = 64
    opened: tuple[int, bytes | None] = (0, None)
    """open's flags and mtp_dir."""
    image_tokens_max: int = 256
    images: dict[int, list[Image]] = field(default_factory=lambda: {})
    """Each slot's images, as admit_images gave them."""
    cards: int = 2
    npp: int = 0
    kv_pages: int = 0
    table: dict[int, list[int]] = field(default_factory=lambda: {})
    """Each slot's mapped pages, as the plans run so far left them."""
    queued_table: dict[int, int] = field(default_factory=lambda: {})
    """Each slot's mapped page count once every queued plan has run."""
    cells: dict[int, list[int | None]] = field(default_factory=lambda: {})
    """Each page's 4 cells: the token written at each position."""
    page_plans: int = 0
    """Plans that carried page changes."""
    config: abi.CConfig | None = None
    asleep: bool = False
    error: str = ""
    """What the next ERROR event reports through last_error."""

    def open(self, config: abi.CConfig) -> abi.CInfo:
        self.opened = (config.flags, config.mtp_dir)
        self.config = config
        return self._info(config)

    def _info(self, config: abi.CConfig) -> abi.CInfo:
        info = abi.CInfo()
        info.api_version = abi.ABI_VERSION
        info.cards = 2
        info.slots = config.slots
        info.max_ctx = config.max_ctx
        info.prompt_chunk = self.prompt_chunk
        info.checkpoints = config.checkpoints
        info.max_step_rows = abi.MAX_STEP_SLOTS
        info.image_tokens_max = self.image_tokens_max
        self.max_ctx = config.max_ctx
        self.npp = -(-config.max_ctx // 4 // self.cards) * self.cards
        self.kv_pages = -(-config.kv_pages // self.cards) * self.cards if config.kv_pages else config.slots * self.npp
        info.kv_pages = self.kv_pages
        info.pages_per_slot = self.npp
        return info

    def admit(self, slot: int, settings: abi.CSettings, prompt: Sequence[int]) -> None:
        assert slot not in self.slots, "admitted into a used slot"
        assert all(
            slot not in [p.slot for p in q.pieces[: q.npieces]] + list(q.slots[: q.nslots]) + [c.slot for c in pages]
            for q, pages in self.plans
        ), "admitted while a queued plan names the slot"
        self.slots[slot] = _Slot(abi.CSettings.from_buffer_copy(settings), list(prompt))
        self.table[slot] = []
        self.queued_table[slot] = 0

    def admit_images(self, slot: int, images: Sequence[Image]) -> None:
        assert slot in self.slots, "images for a slot not admitted"
        assert not self.slots[slot].tokens, "images after the slot began"
        self.images[slot] = list(images)

    def release(self, slot: int) -> None:
        self.slots.pop(slot, None)
        self.images.pop(slot, None)

    def submit(self, plan: abi.CStepPlan) -> bool:
        if len(self.plans) >= self.queue_limit:
            return False
        pages = [abi.CPageChange.from_buffer_copy(plan.pages[i]) for i in range(plan.npages)]
        for c in pages:
            mapped = self.queued_table.get(c.slot, 0)
            assert c.logical <= mapped < self.npp + 1, f"logical page {c.logical} out of order ({mapped} mapped)"
            if c.page == abi.NO_PAGE:
                self.queued_table[c.slot] = c.logical
                continue
            assert c.page < self.kv_pages, f"page {c.page} past the pool"
            assert c.page % self.cards == c.logical % self.cards, "page on the wrong card"
            assert c.cells < 4
            assert not c.cells or c.source % self.cards == c.page % self.cards, "copied from the wrong card"
            if c.logical == mapped:
                self.queued_table[c.slot] = mapped + 1
        self.plans.append((abi.CStepPlan.from_buffer_copy(plan), pages))
        return True

    def _mapped(self, slot: int) -> int:
        return 4 * len(self.table[slot])

    def _stall(self, slot: int, s: _Slot) -> None:
        if not s.stalled:
            s.stalled = True
            self._emit(abi.CEventKind.STALLED, slot, -1, len(s.tokens))

    def _write(self, slot: int, s: _Slot, tokens: Sequence[int]) -> None:
        for t in tokens:
            pos = len(s.tokens)
            self.cells.setdefault(self.table[slot][pos // 4], [None] * 4)[pos % 4] = t
            s.tokens.append(t)

    def _read(self, slot: int, n: int) -> list[int]:
        """The slot's first n positions, read back through its page table, as attention reads them."""
        out = [self.cells[self.table[slot][pos // 4]][pos % 4] for pos in range(n)]
        assert None not in out, f"slot {slot} reads a cell never written"
        return [t for t in out if t is not None]

    def _apply(self, pages: Sequence[abi.CPageChange]) -> None:
        self.page_plans += bool(pages)
        for c in pages:
            table = self.table[c.slot]
            if c.page == abi.NO_PAGE:
                del table[c.logical :]
                continue
            if c.cells:
                self.cells.setdefault(c.page, [None] * 4)[: c.cells] = self.cells[c.source][: c.cells]
            if c.logical == len(table):
                table.append(c.page)
            else:
                table[c.logical] = c.page
            if c.slot in self.slots:
                self.slots[c.slot].stalled = False

    def _emit(self, kind: abi.CEventKind, slot: int, token: int, value: int) -> None:
        self.events.append(abi.CEvent(kind, slot, token, value))

    def _produce(self, slot: int, s: _Slot) -> None:
        tok = next_token(self._read(slot, len(s.tokens)))
        s.next = tok
        s.generated += 1
        self._emit(abi.CEventKind.TOKEN, slot, tok, len(s.tokens))
        stop = tok in list(s.settings.stop[: s.settings.n_stop])
        if stop or s.generated >= s.settings.max_tokens:
            s.done = True
            self._emit(abi.CEventKind.DONE, slot, tok, abi.CDone.STOP if stop else abi.CDone.LENGTH)

    def _run(self, plan: abi.CStepPlan, pages: Sequence[abi.CPageChange]) -> None:
        self.seq += 1
        self._apply(pages)
        # The core may lend the pool's top pages to the expert cache: no slot the plan runs maps one.
        for slot in {p.slot for p in plan.pieces[: plan.npieces]} | set(plan.slots[: plan.nslots]):
            s = self.slots.get(slot)
            if s is not None and not s.done:
                held = self.table.get(slot, [])
                assert all(p < self.kv_pages - plan.kv_free for p in held), "a slot maps a page the plan calls free"
        pieces = [(p.slot, p.tokens) for p in plan.pieces[: plan.npieces]]
        self.log.append((pieces, list(plan.slots[: plan.nslots])))
        for slot, n in pieces:
            s = self.slots.get(slot)
            if s is None or s.done or len(s.tokens) >= len(s.prompt):
                continue
            pos = len(s.tokens)
            if self._mapped(slot) <= pos:
                self._stall(slot, s)
                continue
            n = min(n or self.prompt_chunk, self.prompt_chunk, len(s.prompt) - pos, self._mapped(slot) - pos)
            self._write(slot, s, s.prompt[pos : pos + n])
            self.timing.append(abi.CTiming(abi.CTimingKind.PROMPT, self.seq, n, 0, 0, 1000 * n))
            self._emit(abi.CEventKind.PROMPT, slot, -1, len(s.tokens))
            if len(s.tokens) == len(s.prompt):
                self._produce(slot, s)
        rows = 0
        for slot in plan.slots[: plan.nslots]:
            s = self.slots.get(slot)
            if s is None or s.done:
                continue
            assert len(s.tokens) >= len(s.prompt), "the core would read the prompt one token per step"
            if self._mapped(slot) <= len(s.tokens):
                self._stall(slot, s)
                continue
            self._write(slot, s, [s.next])
            rows += 1
            s.steps += 1
            self._produce(slot, s)
        if rows:
            self.timing.append(abi.CTiming(abi.CTimingKind.DECODE, self.seq, rows, 0, 0, 10_000_000))
            self.decode_steps += 1
            self.decode_rows += rows
        self._emit(abi.CEventKind.STEP, 0, -1, self.seq)

    def _drain(self) -> None:
        while self.plans:
            self._run(*self.plans.popleft())

    def poll(self, cap: int) -> list[abi.CEvent]:
        if not self.events and self.plans:
            self._run(*self.plans.popleft())
        out: list[abi.CEvent] = []
        while self.events and len(out) < cap:
            out.append(self.events.popleft())
        return out

    def wait(self, timeout_ms: int) -> bool:
        return bool(self.events or self.plans)

    def checkpoint(self, slot: int, checkpoint_id: int) -> None:
        self._drain()
        self.checkpoints[checkpoint_id] = list(self.slots[slot].tokens)

    def restore(self, slot: int, checkpoint_id: int) -> None:
        self._drain()
        s = self.slots[slot]
        ck = self.checkpoints[checkpoint_id]
        assert not s.tokens
        assert len(ck) < len(s.prompt)
        assert s.prompt[: len(ck)] == ck
        s.tokens = list(ck)

    def drop(self, checkpoint_id: int) -> None:
        self.checkpoints.pop(checkpoint_id, None)

    def read_timing(self, cap: int) -> list[abi.CTiming]:
        out: list[abi.CTiming] = []
        while self.timing and len(out) < cap:
            out.append(self.timing.popleft())
        return out

    def read_stats(self) -> abi.CStats:
        return abi.CStats(up_ns=1_000_000_000, plans=self.seq, steps=self.decode_steps, step_rows=self.decode_rows)

    def slot_stats(self, slot: int) -> abi.CSlotStats:
        s = self.slots.get(slot)
        if s is None:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, "slot_stats: the slot is not admitted")
        return abi.CSlotStats(steps=s.steps)

    def sleep(self) -> None:
        assert not self.slots, "asleep with a slot admitted"
        assert not self.plans, "asleep with a plan queued"
        self.checkpoints.clear()
        self.table.clear()
        self.queued_table.clear()
        self.cells.clear()
        self.asleep = True

    def wake(self) -> abi.CInfo:
        assert self.asleep
        assert self.config is not None
        self.asleep = False
        return self._info(self.config)

    def last_error(self) -> str:
        return self.error

    def close(self) -> None:
        pass


def expected(prompt: Sequence[int], n: int) -> list[int]:
    tokens = list(prompt)
    out: list[int] = []
    for _ in range(n):
        out.append(next_token(tokens))
        tokens.append(out[-1])
    return out


@dataclass
class Run:
    core: FakeCore
    engine: NativeEngine
    scheduler: Scheduler
    events: list[protocol.Event] = field(default_factory=lambda: [])

    def add(self, rid: int, prompt: Sequence[int], max_new: int, images: tuple[Image, ...] = ()) -> None:
        gen = protocol.Gen(rid, max_new, tuple(prompt), protocol.Settings(temperature=0))
        self.events += self.scheduler.add(gen, images=images)

    def until_idle(self, limit: int = 10_000) -> None:
        for _ in range(limit):
            if self.scheduler.idle:
                return
            self.events += self.scheduler.tick(0)
        raise AssertionError("the scheduler did not finish")

    def tokens(self, rid: int) -> list[int]:
        return [t for e in self.events if isinstance(e, protocol.Tokens) and e.rid == rid for t in e.ids]

    def done(self, rid: int) -> protocol.Done:
        (d,) = [e for e in self.events if isinstance(e, protocol.Done) and e.rid == rid]
        return d


def make(
    config: OpenConfig | None = None,
    rule: CheckpointRule = every_16k,
    *,
    prompt_chunk: int = 64,
    queue_limit: int = 64,
    options: SchedulerOptions | None = None,
) -> Run:
    fake = FakeCore(prompt_chunk=prompt_chunk, queue_limit=queue_limit)
    engine = NativeEngine(core=fake)
    config = replace(config or open_config(), shard_paths=("model.gguf",))
    facts = engine.open(config)
    return Run(fake, engine, Scheduler(engine, facts, options, rule))


def prompt_of(n: int, seed: int = 0) -> list[int]:
    return [3 + (seed * 7919 + i * 104729) % (VOCAB - 3) for i in range(n)]


def test_one_request_decodes_to_its_length():
    run = make()
    prompt = prompt_of(40)
    run.add(1, prompt, 12)
    run.until_idle()
    assert run.tokens(1) == expected(prompt, 12)
    assert run.done(1).finish is protocol.Finish.LENGTH
    assert not [e for e in run.events if isinstance(e, protocol.Error)]


def test_a_long_prompt_goes_in_pieces_of_the_core_chunk_and_never_token_by_token():
    run = make(prompt_chunk=64)
    prompt = prompt_of(300)
    run.add(1, prompt, 5)
    run.until_idle()
    assert run.tokens(1) == expected(prompt, 5)
    pieces = [n for plan_pieces, _ in run.core.log for _, n in plan_pieces]
    assert sum(pieces) == 300
    assert max(pieces) <= 64
    assert all(len(p) <= abi.MAX_PROMPT_PIECES for p, _ in run.core.log)


def test_two_requests_decode_in_the_same_steps():
    run = make()
    a, b = prompt_of(20, 1), prompt_of(30, 2)
    run.add(1, a, 20)
    run.add(2, b, 20)
    run.until_idle()
    assert run.tokens(1) == expected(a, 20)
    assert run.tokens(2) == expected(b, 20)
    assert any(len(slots) == 2 for _, slots in run.core.log)


def test_eos_ends_a_request():
    run = make()
    prompt = prompt_of(10)
    first = expected(prompt, 3)
    run.add(1, prompt, 50)
    run.engine._stop_ids = (first[2],)  # pyright: ignore[reportPrivateUsage]
    run.until_idle()
    assert run.tokens(1) == first
    assert run.done(1).finish is protocol.Finish.STOP


def test_a_stopped_request_gets_no_more_tokens():
    run = make()
    run.add(1, prompt_of(10), 1000)
    for _ in range(20):
        run.events += run.scheduler.tick(0)
    run.events += run.scheduler.cancel(1)
    seen = len(run.tokens(1))
    run.until_idle()
    assert run.done(1).finish is protocol.Finish.CANCEL
    assert len(run.tokens(1)) == seen


def test_a_slot_used_again_after_a_request_ends():
    run = make(open_config(slots=1))
    a, b = prompt_of(12, 1), prompt_of(9, 2)
    run.add(1, a, 30)
    run.add(2, b, 7)
    run.until_idle()
    assert run.tokens(1) == expected(a, 30)
    assert run.tokens(2) == expected(b, 7)


def test_a_shared_prefix_is_restored_from_a_checkpoint():
    def at_64(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
        return [(64, True)] if len(prompt) > 64 else []

    run = make(rule=at_64)
    prefix = prompt_of(64, 5)
    a, b = [*prefix, *prompt_of(10, 6)], [*prefix, *prompt_of(7, 7)]
    run.add(1, a, 6)
    run.until_idle()
    run.add(2, b, 6)
    run.until_idle()
    assert run.tokens(1) == expected(a, 6)
    assert run.tokens(2) == expected(b, 6)
    assert run.done(2).reused == 64


def test_a_prefix_ending_inside_a_page_shares_the_full_pages_and_copies_the_last():
    def at_66(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
        return [(66, False)] if len(prompt) > 66 else []

    run = make(rule=at_66)
    prefix = prompt_of(66, 5)
    a, b = [*prefix, *prompt_of(10, 6)], [*prefix, *prompt_of(7, 7)]
    run.add(1, a, 6)
    run.add(2, b, 6)
    run.until_idle()
    assert run.tokens(1) == expected(a, 6)
    assert run.tokens(2) == expected(b, 6)
    assert run.done(2).reused == 66
    t1, t2 = run.core.table[0], run.core.table[1]
    assert t1[:16] == t2[:16], "the 16 full pages of the prefix are shared"
    assert t1[16] != t2[16], "the 17th, which holds 2 of the prefix's cells, is copied"


def test_decoding_maps_pages_in_batches():
    run = make()
    prompt = prompt_of(10)
    run.add(1, prompt, 600)
    run.until_idle()
    assert run.tokens(1) == expected(prompt, 600)
    assert len(run.core.table[0]) == 153, "610 cells"
    assert run.core.page_plans <= 4, "decode_batch cells at a time"


def test_a_slot_without_pages_stalls_and_resumes():
    run = make(options=SchedulerOptions(decode_lookahead=1, decode_batch=0))
    prompt = prompt_of(8)
    run.add(1, prompt, 40)
    run.until_idle()
    assert run.tokens(1) == expected(prompt, 40)
    assert not [e for e in run.events if isinstance(e, protocol.Error)]


def test_pages_on_two_cards_alternate_by_logical_page():
    run = make()
    run.add(1, prompt_of(50), 30)
    run.until_idle()
    assert all(page % 2 == logical % 2 for logical, page in enumerate(run.core.table[0]))


def test_step_done_carries_the_rows_and_the_time():
    run = make(prompt_chunk=256)
    run.add(1, prompt_of(50), 3)
    run.add(2, prompt_of(100), 3)
    steps: list[tuple[int, int, int, int]] = []
    for _ in range(1000):
        if run.scheduler.idle:
            break
        for e in run.engine.wait(0):
            if e.kind is EventKind.STEP_DONE:
                steps.append((e.a, e.b, e.t_ns, e.c))
            run.scheduler._handle(e, run.events)  # pyright: ignore[reportPrivateUsage]
        run.scheduler._admit(run.events)  # pyright: ignore[reportPrivateUsage]
        run.scheduler._plan()  # pyright: ignore[reportPrivateUsage]
    assert (0, 150, 150_000, 1) in steps, "two pieces, the 50-row one read as windows"
    assert (2, 0, 10_000_000, 0) in steps


def test_done_carries_the_slots_counters_and_the_queue_wait():
    run = make()
    run.add(1, prompt_of(40), 5)
    run.until_idle()
    done = run.done(1)
    assert done.steps == done.generated - 1 == 4, "the first token comes with the prompt, each later one from a step"
    assert done.queue_ms >= 0
    assert (done.drafts, done.accepted) == (0, 0)


def test_the_engines_stats_are_the_cores():
    run = make()
    assert run.engine.stats().plans == 0
    run.add(1, prompt_of(40), 3)
    run.until_idle()
    stats = run.engine.stats()
    assert stats.plans == run.core.seq > 0
    assert stats.steps == run.core.decode_steps >= 2
    assert stats.up_ns == 1_000_000_000
    assert run.engine.slot_stats(7).steps == 0, "zeros for a slot not admitted"


def test_a_full_plan_queue_is_waited_out():
    run = make(queue_limit=1)
    prompt = prompt_of(400)
    run.add(1, prompt, 4)
    run.until_idle()
    assert run.tokens(1) == expected(prompt, 4)


def test_open_refuses_more_slots_than_a_step_decodes():
    engine = NativeEngine(core=FakeCore())
    with pytest.raises(Exception, match="slots"):
        engine.open(replace(open_config(slots=abi.MAX_STEP_SLOTS + 1), shard_paths=("m.gguf",)))


def test_the_facts_fit_the_protocols_info_line():
    engine = NativeEngine(core=FakeCore())
    facts = engine.open(replace(open_config(), shard_paths=("m.gguf",)))
    protocol.Info({"version": facts.version, "cards": str(facts.cards)}).encode()


def test_admit_passes_min_p_and_the_penalties_to_the_core():
    run = make()
    run.engine.admit(0, SlotSettings(0.7, 0.9, 20, 0.05, 64, 1.1, 0.25, 0.5, 7, 4), tuple(prompt_of(8)))
    c = run.core.slots[0].settings
    assert (c.min_p, c.penalty_last_n, c.penalty_repeat, c.penalty_freq, c.penalty_present) == pytest.approx(
        (0.05, 64, 1.1, 0.25, 0.5)
    )


def test_open_asks_for_swaps_and_drafting_but_with_adr_0016s_switches():
    fake = FakeCore()
    NativeEngine(core=fake).open(replace(open_config(), shard_paths=("m.gguf",), draft_dir="/models/mtp"))
    assert fake.opened == (abi.ConfigFlag.SWAPS | abi.ConfigFlag.SHARE_KV, b"/models/mtp")
    fake = FakeCore()
    NativeEngine(core=fake).open(replace(open_config(), shard_paths=("m.gguf",), deterministic=True))
    assert fake.opened == (0, None)


def test_sleep_and_wake_give_the_same_tokens_after():
    run = make()
    prompt = prompt_of(70)
    run.add(1, prompt, 8)
    run.until_idle()
    run.engine.sleep()
    assert run.core.asleep
    with pytest.raises(EngineError, match="asleep"):
        run.engine.admit(0, GREEDY, (1, 2))
    facts = run.engine.wake()
    run.scheduler = Scheduler(run.engine, facts)
    run.add(2, prompt, 8)
    run.until_idle()
    assert run.tokens(2) == run.tokens(1) == expected(prompt, 8)


def test_sleep_refuses_an_admitted_slot():
    run = make()
    run.engine.admit(0, GREEDY, (5, 6, 7))
    with pytest.raises(EngineError, match="admitted"):
        run.engine.sleep()


def test_a_core_error_logs_what_the_core_said(caplog: pytest.LogCaptureFixture):
    run = make()
    run.core.error = "card 1 did not finish a list in 60 s"
    run.core.events.append(abi.CEvent(kind=abi.CEventKind.ERROR, value=1))
    with caplog.at_level("ERROR"):
        events = run.engine.poll()
    assert [e.kind for e in events] == [EventKind.ERROR]
    assert "core error 1: card 1 did not finish a list in 60 s" in caplog.text
