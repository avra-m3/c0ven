from __future__ import annotations

import random

import pytest

from engine.api.interface import PageOp
from engine.seqstate.checkpoints import CheckpointPool
from engine.seqstate.manager import SeqState, Sequence, pages_for
from engine.seqstate.pages import HIGH_BLOCK, PagePool, PoolExhausted


def test_pool_reservations():
    pool = PagePool(4)
    pool.reserve(3)
    assert pool.available == 1
    with pytest.raises(PoolExhausted):
        pool.reserve(2)
    a = pool.take()
    assert (pool.free, pool.reserved, pool.refs(a)) == (3, 2, 1)
    pool.incref(a)
    pool.decref(a)
    assert pool.refs(a) == 1
    pool.decref(a)
    assert pool.free == 4
    with pytest.raises(ValueError, match="already free"):
        pool.decref(a)


def test_pages_are_taken_lowest_first_and_high_follows_the_top_one():
    pool = PagePool(4 * HIGH_BLOCK, classes=2)
    assert pool.high == 0
    pool.reserve((HIGH_BLOCK + 1, HIGH_BLOCK + 1))
    pages = [pool.take(i) for i in range(2 * HIGH_BLOCK)]
    assert pages == list(range(2 * HIGH_BLOCK))
    assert pool.high == 2 * HIGH_BLOCK
    for p in pages[1:]:
        pool.decref(p)
    assert pool.high == HIGH_BLOCK
    assert pool.take(1) == 1  # a freed page comes back before any higher one
    pool.decref(pages[0])
    pool.decref(1)
    assert pool.high == 0


def test_take_needs_a_reservation():
    with pytest.raises(PoolExhausted):
        PagePool(2).take()


def admit(s: SeqState, prompt: tuple[int, ...], max_new: int) -> Sequence:
    a = s.plan_admission(prompt, max_new)
    assert a is not None
    return s.admit(a, prompt, max_new)


def test_admission_reserves_the_prompt_and_a_block():
    s = SeqState(slots=2, pages=16, checkpoints=2)
    prompt = tuple(range(10))
    admission = s.plan_admission(prompt, max_new=100, ahead=4)
    assert admission is not None
    assert admission.pages == pages_for(14) == 4
    short = s.plan_admission(prompt, max_new=1, ahead=4)
    assert short is not None
    assert short.pages == pages_for(11), "the block stops at max_new"
    seq = s.admit(admission, prompt, 100)
    assert s.pool.available == 12
    assert s.grow(seq.slot, 14)
    assert (len(seq.pages), seq.reserved) == (4, [0])
    changes = s.take_changes(100)
    assert [c.logical for c in changes] == [0, 1, 2, 3]
    assert all(c.op is PageOp.ASSIGN for c in changes)
    s.release(seq.slot)
    assert (s.pool.free, s.pool.reserved, s.free_slots) == (16, 0, 2)


@pytest.mark.parametrize("classes", [1, 2])
def test_growth_reserves_a_block_at_a_time(classes: int):
    s = SeqState(slots=2, pages=16, checkpoints=2, classes=classes)
    prompt = tuple(range(10))
    a = s.plan_admission(prompt, 100, ahead=2)
    assert a is not None
    seq = s.admit(a, prompt, 100)
    assert s.grow(seq.slot, 12)
    assert s.pool.available == 13
    assert s.grow(seq.slot, 40), "pages past the admission are reserved as the slot grows"
    assert (len(seq.pages), s.pool.available, sum(seq.reserved)) == (10, 6, 0)
    assert not s.grow(seq.slot, 100), "a block the pool cannot give changes nothing"
    assert len(seq.pages) == 10
    assert s.grow(seq.slot, 64)
    assert [p % classes for p in seq.pages] == [i % classes for i in range(16)]
    s.release(seq.slot)
    assert (s.pool.free, s.pool.reserved) == (16, 0)


def test_growth_evicts_only_checkpoints_that_free_pages():
    s = SeqState(slots=3, pages=12, checkpoints=4)
    a = admit(s, tuple(range(16)), 8)
    s.grow(a.slot, 16)
    shared = s.checkpoint(a.slot, 8, kept=False)  # its pages are a's too: evicting it frees none
    c = admit(s, tuple(range(100, 116)), 8)
    s.grow(c.slot, 16)
    alone = s.checkpoint(c.slot, 16, kept=False)
    assert shared is not None
    assert alone is not None
    shared.ready = alone.ready = True
    s.checkpoints.touch(alone)  # least recent use alone would pick `shared`
    s.release(c.slot)
    s.take_changes(100)
    b = admit(s, tuple(range(200, 216)), 8)
    s.grow(b.slot, 16)
    assert s.pool.available == 0
    alone.users.add(c.slot)
    assert not s.grow(b.slot, 20), "a checkpoint being restored from stays"
    assert s.drops == []
    alone.users.clear()
    assert s.grow(b.slot, 20)
    assert s.drops == [alone.id]
    assert list(s.checkpoints) == [shared]


def test_no_slot_no_admission():
    s = SeqState(slots=1, pages=16, checkpoints=1)
    a = s.plan_admission((1, 2), 2)
    assert a is not None
    s.admit(a, (1, 2), 2)
    assert s.plan_admission((3, 4), 2) is None


def test_resume_shares_full_pages_and_copies_the_partial_one():
    s = SeqState(slots=2, pages=32, checkpoints=2)
    p1 = tuple(range(20))
    a = s.plan_admission(p1, 4)
    assert a is not None
    first = s.admit(a, p1, 4)
    s.grow(first.slot, 20)
    s.take_changes(100)
    ck = s.checkpoint(first.slot, 10, kept=False)
    assert ck is not None
    assert ck.pages == tuple(first.pages[:3])
    ck.ready = True

    p2 = (*range(10), 99, 98)
    b = s.plan_admission(p2, 4, ahead=4)
    assert b is not None
    assert b.checkpoint is ck
    assert b.pages == pages_for(16) - 2, "two full pages are shared"
    second = s.admit(b, p2, 4)
    assert second.start == 10
    assert second.pages[:2] == first.pages[:2]
    changes = s.take_changes(100)
    assert [(c.op, c.logical) for c in changes] == [(PageOp.ASSIGN, 0), (PageOp.ASSIGN, 1), (PageOp.COPY, 2)]
    copy = changes[2]
    assert (copy.source, copy.cells) == (first.pages[2], 2)
    assert s.pool.refs(first.pages[0]) == 3, "the producer, the checkpoint and the new sequence"

    s.release(first.slot)
    s.release(second.slot)
    assert s.pool.refs(first.pages[0]) == 1, "the checkpoint keeps its pages"
    s.checkpoints.remove(ck.id)
    assert s.pool.free == 32


def test_checkpoints_are_evicted_least_recently_used_kept_last():
    pool = PagePool(8)
    cps = CheckpointPool(2, pool, max_kept=1)
    kept, _ = cps.add((1,), (), kept=True)
    other, _ = cps.add((2,), (), kept=False)
    kept.ready = other.ready = True
    cps.touch(kept)
    third, evicted = cps.add((3,), (), kept=False)
    assert evicted == [other.id]
    assert {c.tokens for c in cps} == {(1,), (3,)}
    third.ready = True
    assert cps.victim() is third


def _conversation(s: SeqState, system: int, turns: int, turn: int, first: int) -> tuple[int, ...]:
    """A conversation of `turns` requests, each `turn` tokens longer, checkpointed where its last turn starts and at
    the end of its system prompt (kept), as qwen4exp's rule does. Returns its last prompt."""
    hist = list(range(first, first + system))
    for t in range(turns):
        hist += range(first + system + t * turn, first + system + (t + 1) * turn)
        prompt = tuple(hist)
        a = s.plan_admission(prompt, 0)
        assert a is not None
        seq = s.admit(a, prompt, 0)
        s.grow(seq.slot, len(prompt))
        s.take_changes(len(prompt))
        s.submitted()
        for position, kept in ((system, True), (len(prompt) - turn, False)):
            if seq.start < position < len(prompt):
                ck = s.checkpoint(seq.slot, position, kept)
                if ck is not None:
                    ck.ready = True
        s.release(seq.slot)
    return tuple(hist)


def test_a_pool_with_nothing_to_evict_refuses_and_changes_nothing():
    s = SeqState(slots=1, pages=16, checkpoints=1)
    a = s.plan_admission((1, 2, 3, 4), 2)
    assert a is not None
    seq = s.admit(a, (1, 2, 3, 4), 2)
    s.grow(seq.slot, 4)
    held = s.checkpoint(seq.slot, 2, kept=True)
    assert held is not None
    assert not held.ready
    assert s.checkpoint(seq.slot, 3, kept=True) is None
    assert s.drops == []
    assert list(s.checkpoints) == [held]
    assert held.kept, "the refused one demotes no kept checkpoint"
    assert not CheckpointPool(0, s.pool).has_room()


def test_checkpoints_stay_spread_back_from_the_end():
    s = SeqState(slots=1, pages=1024, checkpoints=8)
    last = _conversation(s, system=96, turns=200, turn=10, first=0)
    back = sorted(len(last) - len(c.tokens) for c in s.checkpoints)
    assert back[0] == 10, "the turn before is kept"
    assert back[-1] == len(last) - 96, "the system prompt is kept"
    assert back[-2] < (len(last) - 96) / 2, "the others thin out towards the start"
    for edit in (60, 420):
        edited = last[: len(last) - edit] + (-1,) * edit
        ck = s.checkpoints.longest_prefix(edited)
        assert ck is not None
        assert len(edited) - len(ck.tokens) < 2 * edit, f"an edit {edit} back reads less than twice that again"


def test_a_paused_conversations_newest_checkpoint_outlives_another_ones_older_ones():
    s = SeqState(slots=1, pages=1024, checkpoints=4)
    paused = _conversation(s, system=8, turns=3, turn=10, first=0)
    _conversation(s, system=8, turns=6, turn=10, first=10_000)
    ck = s.checkpoints.longest_prefix((*paused, 1))
    assert ck is not None
    assert len(ck.tokens) == len(paused) - 10, "it resumes where its last turn started"


def test_parents_follow_adds_and_removes():
    pool = PagePool(8)
    cps = CheckpointPool(8, pool)
    a, _ = cps.add((1, 2), (), False)
    c, _ = cps.add((1, 2, 3, 4), (), False)
    other, _ = cps.add((1, 9), (), False)
    assert (a.parent, c.parent, other.parent) == (None, a, None)
    b, _ = cps.add((1, 2, 3), (), False)
    assert (b.parent, c.parent) == (a, b), "a checkpoint between two becomes the longer one's parent"
    cps.remove(b.id)
    assert c.parent is a
    a.ready = c.ready = other.ready = True
    assert cps.spread_victim() is a
    cps.remove(c.id)
    assert cps.spread_victim() is cps.victim(), "with none extended, least recently used"


def test_longest_prefix_is_strict_and_ready():
    pool = PagePool(8)
    cps = CheckpointPool(4, pool)
    short, _ = cps.add((1, 2), (), False)
    long, _ = cps.add((1, 2, 3), (), False)
    whole, _ = cps.add((1, 2, 3, 4), (), False)
    assert cps.longest_prefix((1, 2, 3, 4)) is None, "nothing is ready"
    short.ready = long.ready = whole.ready = True
    assert cps.longest_prefix((1, 2, 3, 4)) is long, "a checkpoint of the whole prompt leaves nothing to compute"
    assert cps.longest_prefix((1, 2, 9)) is short
    assert cps.longest_prefix((9,)) is None


def test_admission_evicts_checkpoints_for_room():
    s = SeqState(slots=2, pages=8, checkpoints=2)
    a = s.plan_admission(tuple(range(16)), 0)
    assert a is not None
    seq = s.admit(a, tuple(range(16)), 0)
    s.grow(seq.slot, 16)
    ck = s.checkpoint(seq.slot, 16, kept=False)
    assert ck is not None
    ck.ready = True
    s.release(seq.slot)
    assert s.pool.available == 4
    b = s.plan_admission((50, 51, 52, 53) * 8, 0)
    assert b is not None
    assert s.drops == [ck.id]
    assert s.pool.available == 8


def _turn(s: SeqState, prompt: tuple[int, ...], at: int) -> int:
    """Admit `prompt`, read it and checkpoint it after `at` tokens. Returns the slot, still admitted."""
    a = s.plan_admission(prompt, 0)
    assert a is not None
    seq = s.admit(a, prompt, 0)
    assert s.grow(seq.slot, len(prompt))
    s.take_changes(len(prompt))
    s.submitted()
    ck = s.checkpoint(seq.slot, at, kept=False)
    assert ck is not None
    ck.ready = True
    return seq.slot


def test_admission_keeps_the_checkpoints_of_a_conversation_in_a_slot():
    s = SeqState(slots=3, pages=12, checkpoints=4)
    writing = tuple(range(16))
    slot = _turn(s, writing, 12)  # the least recently used checkpoint; the slot holds its pages
    waiting = tuple(range(100, 116))
    s.release(_turn(s, waiting, 12))
    assert s.pool.available == 5
    a = s.plan_admission(tuple(range(200, 224)), 0)
    assert a is not None
    assert [len(c.tokens) for c in s.checkpoints] == [12]
    assert s.checkpoints.longest_prefix(writing) is not None, "evicting it would have freed no page"
    s.admit(a, tuple(range(200, 224)), 0)
    s.release(slot)
    again = s.plan_admission((*writing, 16, 17), 0)
    assert again is not None
    assert again.checkpoint is not None


def test_admission_takes_a_waiting_conversation_from_its_end():
    s = SeqState(slots=2, pages=12, checkpoints=4)
    history = tuple(range(32))
    for at in (8, 16, 24):
        s.release(_turn(s, history[: at + 8], at))
    assert [len(c.tokens) for c in s.checkpoints] == [8, 16, 24]
    assert s.pool.available == 6
    a = s.plan_admission(tuple(range(100, 132)), 0)
    assert a is not None
    assert [len(c.tokens) for c in s.checkpoints] == [8, 16], "the older ones hold no page of their own"
    best = s.checkpoints.longest_prefix(history)
    assert best is not None
    assert len(best.tokens) == 16


def test_checkpoints_that_end_on_the_same_page_are_evicted_together():
    s = SeqState(slots=2, pages=4, checkpoints=4)
    prompt = tuple(range(16))
    slot = _turn(s, prompt, 13)
    twin = s.checkpoint(slot, 15, kept=False)  # the same four pages
    assert twin is not None
    twin.ready = True
    s.release(slot)
    assert s.pool.available == 0
    assert s.plan_admission(tuple(range(100, 108)), 0) is not None
    assert len(s.checkpoints) == 0


@pytest.mark.parametrize("seed", range(20))
def test_only_a_checkpoints_last_pages_can_be_its_alone(seed: int):
    rng = random.Random(seed)
    k = 2
    s = SeqState(slots=3, pages=64, checkpoints=6, classes=k)
    prompts: list[tuple[int, ...]] = []
    slots: list[int] = []
    for step in range(200):
        if slots and (len(slots) == 3 or rng.random() < 0.4):
            s.release(slots.pop(rng.randrange(len(slots))))
            continue
        base = rng.choice(prompts)[: rng.randrange(1, 40)] if prompts and rng.random() < 0.7 else ()
        prompt = base + tuple(range(1000 * step, 1000 * step + rng.randrange(1, 24)))
        a = s.plan_admission(prompt, 0)
        s.drops.clear()
        if a is None:
            continue
        seq = s.admit(a, prompt, 0)
        if not s.grow(seq.slot, len(prompt)):
            s.release(seq.slot)
            continue
        s.take_changes(1000)
        s.submitted()
        slots.append(seq.slot)
        prompts.append(prompt)
        for at in sorted(rng.sample(range(1, len(prompt) + 1), min(2, len(prompt)))):
            if at > seq.start and (ck := s.checkpoint(seq.slot, at, kept=False)) is not None:
                ck.ready = True
        for ck in s.checkpoints:
            for c in range(k):
                alone = any(s.pool.refs(p) == 1 and p % k == c for p in ck.pages)
                at_end = any(s.pool.refs(p) == 1 and p % k == c for p in ck.pages[-k:])
                assert alone == at_end
