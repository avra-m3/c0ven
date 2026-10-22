from __future__ import annotations

import pytest

from engine.api.interface import PageOp
from engine.seqstate.checkpoints import CheckpointPool
from engine.seqstate.manager import SeqState, pages_for
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


def test_admission_reserves_the_whole_sequence():
    s = SeqState(slots=2, pages=16, checkpoints=2)
    prompt = tuple(range(10))
    admission = s.plan_admission(prompt, max_new=6)
    assert admission is not None
    assert admission.pages == pages_for(16) == 4
    seq = s.admit(admission, prompt, 6)
    assert s.pool.available == 12
    s.grow(seq.slot, 10)
    assert len(seq.pages) == 3
    s.grow(seq.slot, 1000)
    assert len(seq.pages) == 4, "growth stops at the reservation"
    changes = s.take_changes(100)
    assert [c.logical for c in changes] == [0, 1, 2, 3]
    assert all(c.op is PageOp.ASSIGN for c in changes)
    s.release(seq.slot)
    assert (s.pool.free, s.pool.reserved, s.free_slots) == (16, 0, 2)


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
    b = s.plan_admission(p2, 4)
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
