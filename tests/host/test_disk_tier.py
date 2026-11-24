"""The disk tier (engine/seqstate/disk.py) with the scheduler and the mock engine, whose files hold each page's cells:
a checkpoint read back from disk gives the same tokens as one computed from the start."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import protocol
from engine.api.mock import MockEngine
from engine.scheduler import CheckpointRule, Scheduler
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
    pages: int | None = None,
    checkpoints: int = 4,
    cap: int = 2**30,
) -> Run:
    engine = MockEngine(history)
    facts = engine.open(open_config(slots=slots, context=8192, pages=pages, checkpoints=checkpoints))
    disk = DiskTier(tmp, "t", cap) if tmp is not None else None
    return Run(engine, Scheduler(engine, facts, checkpoint_rule=rule, disk=disk))


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
