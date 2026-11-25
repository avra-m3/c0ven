"""The disk tier (engine/seqstate/disk.py) with the scheduler and the mock engine, whose files hold each page's cells:
a checkpoint read back from disk gives the same tokens as one computed from the start."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import protocol
from engine.api.mock import MockEngine
from engine.scheduler import CheckpointRule, Scheduler, SchedulerOptions
from engine.seqstate.disk import BLOCK, MIN_TOKENS, DiskTier, chain_of, storable
from tests.helpers import open_config


def history(tokens: Sequence[int]) -> int:
    """A token from the position and the tokens before it, never EOS."""
    return 3 + (len(tokens) * 31 + tokens[-1] * 7 + tokens[-2]) % 900


def prompt(n: int, start: int = 3) -> tuple[int, ...]:
    return tuple(start + i % 900 for i in range(n))


def marks_at(*positions: int) -> CheckpointRule:
    return lambda p: [(x, False) for x in positions if x < len(p)]


@dataclass
class Run:
    engine: MockEngine
    scheduler: Scheduler
    events: list[protocol.Event] = field(default_factory=lambda: [])

    def add(self, rid: int, tokens: tuple[int, ...], max_new: int) -> None:
        self.events += self.scheduler.add(protocol.Gen(rid, max_new, tokens, protocol.Settings(temperature=0.0)))

    def until_idle(self) -> None:
        for _ in range(100_000):
            if self.scheduler.idle:
                return
            self.events += self.scheduler.tick(0)
        raise AssertionError("the scheduler did not finish")

    def tokens(self, rid: int) -> list[int]:
        return [t for e in self.events if isinstance(e, protocol.Tokens) and e.rid == rid for t in e.ids]

    def done(self, rid: int) -> protocol.Done:
        (d,) = [e for e in self.events if isinstance(e, protocol.Done) and e.rid == rid]
        return d

    def errors(self) -> list[protocol.Error]:
        return [e for e in self.events if isinstance(e, protocol.Error)]


def make(
    tmp: Path | None,
    *,
    rule: CheckpointRule,
    slots: int = 1,
    context: int = 8192,
    pages: int | None = None,
    checkpoints: int = 4,
    cap: int = 2**30,
    options: SchedulerOptions | None = None,
) -> Run:
    engine = MockEngine(history)
    facts = engine.open(open_config(slots=slots, context=context, pages=pages, checkpoints=checkpoints))
    disk = DiskTier(tmp, "t", cap) if tmp is not None else None
    return Run(engine, Scheduler(engine, facts, options=options, checkpoint_rule=rule, disk=disk))


A = prompt(1300, start=50)
"""A conversation whose checkpoint at 1,200 tokens holds a whole block and a tail."""


def follow_up() -> tuple[int, ...]:
    return (*A[:1200], 5, 6, 7)


def fresh_tokens() -> list[int]:
    run = make(None, rule=marks_at(1200))
    run.add(9, follow_up(), 20)
    run.until_idle()
    return run.tokens(9)


def test_a_checkpoint_evicted_for_pages_comes_back_from_disk(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200), pages=512)
    run.add(1, A, 1)
    run.until_idle()
    run.add(2, prompt(200), 1500)  # its pages need the checkpoint's
    run.until_idle()
    assert len(run.scheduler.state.checkpoints) == 0
    assert run.scheduler.disk is not None
    assert run.scheduler.disk.stats().saved == 1
    run.add(3, follow_up(), 20)
    run.until_idle()
    assert run.errors() == []
    assert run.done(3).reused == 1200
    assert run.tokens(3) == fresh_tokens()
    assert run.scheduler.disk.stats().loaded == 1


def test_a_checkpoint_evicted_for_the_pools_count_comes_back_from_disk(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200), checkpoints=1)
    run.add(1, A, 1)
    run.until_idle()
    run.add(2, prompt(1300, start=97), 1)  # its checkpoint takes the pool's one place
    run.until_idle()
    run.add(3, follow_up(), 20)
    run.until_idle()
    assert run.errors() == []
    assert run.done(3).reused == 1200
    assert run.tokens(3) == fresh_tokens()


def test_entries_outlive_the_engine_and_the_scheduler(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200))
    run.add(1, A, 1)
    run.until_idle()
    assert run.scheduler.write_all() == 1
    assert run.scheduler.write_all() == 0, "written once"
    again = make(tmp_path, rule=marks_at(1200))
    again.add(3, follow_up(), 20)
    again.until_idle()
    assert again.errors() == []
    assert again.done(3).reused == 1200
    assert again.tokens(3) == fresh_tokens()


def test_a_conversations_checkpoints_share_their_blocks(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1100, 2100))
    run.add(1, prompt(2200), 1)
    run.until_idle()
    assert run.scheduler.write_all() == 2
    assert len(list((tmp_path / "t").glob("b-*.kv"))) == 2, "block 0 is written once, for both"
    run.add(2, (*prompt(2200)[:2100], 9, 9), 1)
    run.until_idle()
    assert run.done(2).reused == 2100


def test_idle_time_writes_ahead_so_an_eviction_writes_nothing(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200), pages=512)
    run.add(1, A, 1)
    run.until_idle()
    assert run.scheduler.write_ahead() == 1
    assert run.scheduler.write_ahead() == 0
    run.add(2, prompt(200), 1500)
    run.until_idle()
    assert run.scheduler.disk is not None
    assert run.scheduler.disk.stats().saved == 1


def test_the_cap_removes_the_entries_used_longest_ago(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200))
    run.add(1, A, 1)
    run.until_idle()
    run.scheduler.write_all()
    assert run.scheduler.disk is not None
    one = run.scheduler.disk.total
    small = make(tmp_path, rule=marks_at(1200), cap=one + one // 2)
    small.add(2, prompt(1300, start=97), 1)
    small.until_idle()
    small.scheduler.write_all()
    assert small.scheduler.disk is not None
    d = small.scheduler.disk.stats()
    assert (d.entries, d.evicted) == (1, 1)
    small.add(3, follow_up(), 2)
    small.until_idle()
    assert small.done(3).reused == 0, "the first conversation's entry went"


def test_a_file_the_engine_refuses_takes_its_entry_and_the_prompt_is_read_whole(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200))
    run.add(1, A, 1)
    run.until_idle()
    run.scheduler.write_all()
    (state,) = (tmp_path / "t").glob("c-*.state")
    state.write_text("{}")
    again = make(tmp_path, rule=marks_at(1200))
    again.add(3, follow_up(), 20)
    again.until_idle()
    assert again.errors() == []
    assert again.done(3).reused == 0
    assert again.tokens(3) == fresh_tokens()
    assert again.scheduler.disk is not None
    assert len(again.scheduler.disk) == 0


def test_a_missing_block_file_removes_its_entries_at_open(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200))
    run.add(1, A, 1)
    run.until_idle()
    run.scheduler.write_all()
    for f in (tmp_path / "t").glob("b-*.kv"):
        f.unlink()
    tier = DiskTier(tmp_path, "t", 2**30)
    assert len(tier) == 0
    assert list((tmp_path / "t").iterdir()) == []


def test_short_prompts_and_images_stay_off_disk():
    assert storable(prompt(MIN_TOKENS))
    assert not storable(prompt(MIN_TOKENS - 1))
    assert not storable((*prompt(MIN_TOKENS), -5))


def test_a_chain_names_each_block_by_every_token_before_its_end():
    a, b = prompt(3 * BLOCK), (*prompt(BLOCK), *prompt(2 * BLOCK, start=7))
    ca, cb = chain_of(a), chain_of(b)
    assert len(ca) == 3
    assert ca[0] == cb[0]
    assert ca[1] != cb[1]
    assert ca[2] != cb[2]
    assert chain_of(a, 2) == ca[:2]


def test_spread_evictions_within_a_long_prompt_do_not_spill_to_disk(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1100, 1400, 1700), checkpoints=2)
    run.add(1, prompt(2000), 1)
    run.until_idle()
    assert run.errors() == []
    assert run.scheduler.disk is not None
    assert run.scheduler.disk.stats().saved == 0


def test_read_back_waits_until_the_whole_request_fits(tmp_path: Path):
    seed = make(tmp_path, rule=marks_at(1200))
    seed.add(1, A, 1)
    seed.until_idle()
    assert seed.scheduler.write_all() == 1

    long_prompt = (*A[:1200], *prompt(1000, start=50))
    ref = make(None, rule=marks_at())
    ref.add(9, long_prompt, 20)
    ref.until_idle()

    again = make(tmp_path, rule=marks_at(1200), slots=2, pages=700)
    again.add(2, prompt(800, start=10), 100)  # holds 225 of 700 pages, leaving 475 (> 300, < 555)
    again.add(3, long_prompt, 20)  # needs 555 pages total; disk entry is 300 pages
    again.until_idle()
    assert again.errors() == []
    assert again.done(3).reused == 1200
    assert again.tokens(3) == ref.tokens(9)
    assert again.scheduler.disk is not None
    assert again.scheduler.disk.stats().loaded == 1


def test_read_back_shares_whole_blocks_with_an_in_memory_checkpoint(tmp_path: Path):
    p2200 = prompt(2300, start=70)
    follow = (*p2200[:2200], 5, 6, 7)
    ref = make(None, rule=marks_at())
    ref.add(9, follow, 10)
    ref.until_idle()

    run = make(tmp_path, rule=marks_at(1200, 2200), pages=600)
    run.add(1, p2200, 1)
    run.until_idle()
    assert run.scheduler.write_all() == 2
    run.add(2, prompt(100, start=400), 1000)  # evicts the 2,200 checkpoint for pages while 1,200 stays
    run.until_idle()
    assert [len(ck.tokens) for ck in run.scheduler.state.checkpoints] == [1200]
    (best,) = list(run.scheduler.state.checkpoints)
    block0 = tmp_path / "t" / f"b-{chain_of(p2200)[0].hex()}.kv"
    block0.unlink()  # proves read-back shares block 0 with `best` instead of reading it from disk
    run.add(3, follow, 10)
    run.until_idle()
    assert run.errors() == []
    assert run.done(3).reused == 2200
    assert run.tokens(3) == ref.tokens(9)
    ck_2200 = run.scheduler.state.checkpoints.find(p2200[:2200])
    assert ck_2200 is not None
    assert ck_2200.pages[:256] == best.pages[:256]


def test_a_preempted_short_request_refreshes_its_chain_and_finds_disk_entries(tmp_path: Path):
    short_prompt = prompt(900, start=50)
    ref = make(None, rule=marks_at())
    ref.add(1, short_prompt, 400)
    ref.until_idle()
    expected_tokens = ref.tokens(1)

    full = (*short_prompt, *expected_tokens[:310])
    seed = make(tmp_path, rule=marks_at(1200))
    seed.add(1, full, 1)
    seed.until_idle()
    assert seed.scheduler.write_all() == 1

    small = SchedulerOptions(decode_lookahead=8, decode_batch=32)
    again = make(tmp_path, rule=marks_at(1200), slots=2, pages=420, options=small)
    again.scheduler.add(protocol.Gen(2, 400, short_prompt, protocol.Settings(temperature=0.0)), priority=0)
    again.scheduler.add(protocol.Gen(3, 500, prompt(10, start=900), protocol.Settings(temperature=0.0)), priority=1)
    again.until_idle()
    assert again.errors() == []
    assert again.tokens(2) == expected_tokens
    assert again.scheduler.disk is not None
    assert again.scheduler.disk.stats().loaded == 1


def test_a_file_of_another_context_is_refused(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200), context=8192)
    run.add(1, A, 1)
    run.until_idle()
    assert run.scheduler.write_all() == 1
    again = make(tmp_path, rule=marks_at(1200), context=16384)
    again.add(3, follow_up(), 20)
    again.until_idle()
    assert again.errors() == []
    assert again.done(3).reused == 0
    assert again.tokens(3) == fresh_tokens()
    assert again.scheduler.disk is not None
    assert len(again.scheduler.disk) == 0


def test_write_ahead_tries_an_oversized_checkpoint_once(tmp_path: Path):
    run = make(tmp_path, rule=marks_at(1200), cap=10)
    run.add(1, A, 1)
    run.until_idle()
    assert run.scheduler.can_write_ahead
    assert run.scheduler.write_ahead() == 0
    assert not run.scheduler.can_write_ahead
