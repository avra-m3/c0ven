"""Sequence-state bookkeeping: slots, page tables, reservations and checkpoints (DESIGN.md, section 4).

Python's part of the sequence-state manager. It hands out integers and records which belong to whom; the core
holds the bytes. Every change to a slot's page table becomes a `PageChange` in the next step plan.

A sequence admitted from a checkpoint at position p shares the checkpoint's full pages and gets its own copy of the
partial page p // 4 (its first p % 4 cells), since it will write the cells after them.

On two cards logical page i must be a page of class i % 2 (`PagePool`), so reservations are counted per class.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.api.interface import PAGE_CELLS, PageChange, PageOp
from engine.seqstate.checkpoints import Checkpoint, CheckpointPool
from engine.seqstate.pages import PagePool


def pages_for(cells: int) -> int:
    return -(-cells // PAGE_CELLS)


@dataclass(slots=True)
class Sequence:
    slot: int
    prompt: tuple[int, ...]
    max_new: int
    start: int
    """Prompt positions restored from a checkpoint."""
    pages: list[int] = field(default_factory=lambda: [])
    """Logical page i is physical page pages[i]."""
    reserved: list[int] = field(default_factory=lambda: [])
    """Pages per class still promised to this sequence and not yet taken."""
    source: Checkpoint | None = None

    @property
    def cells(self) -> int:
        return len(self.pages) * PAGE_CELLS


@dataclass(frozen=True, slots=True)
class Admission:
    slot: int
    checkpoint: Checkpoint | None
    pages: int
    """Pages to reserve: the whole sequence, less the full pages shared with the checkpoint."""
    need: tuple[int, ...] = ()
    """The same, per class."""


class SeqState:
    def __init__(self, slots: int, pages: int, checkpoints: int, max_kept: int = 2, classes: int = 1) -> None:
        self.pool = PagePool(pages, classes)
        self.checkpoints = CheckpointPool(checkpoints, self.pool, max_kept)
        self.sequences: dict[int, Sequence] = {}
        self._free_slots = list(range(slots - 1, -1, -1))
        self._changes: list[PageChange] = []
        self.drops: list[int] = []
        """Checkpoint ids evicted here; the scheduler passes each to the core's `drop`."""

    @property
    def free_slots(self) -> int:
        return len(self._free_slots)

    def plan_admission(self, prompt: tuple[int, ...], max_new: int) -> Admission | None:
        """Whether a request fits now, evicting checkpoints for room if it must. None if it does not fit."""
        if not self._free_slots:
            return None
        ck = self.checkpoints.longest_prefix(prompt)
        shared = len(ck.tokens) // PAGE_CELLS if ck is not None else 0
        need = self.pool.split(shared, pages_for(len(prompt) + max_new))
        while not self.pool.can_reserve(need):
            victim = self.checkpoints.victim()
            if victim is None or victim is ck:
                others = [c for c in self.checkpoints if c is not ck and not c.users and c.ready]
                if not others:
                    return None
                victim = min(others, key=lambda c: (c.kept, c.used))
            self.checkpoints.remove(victim.id)
            self.drops.append(victim.id)
        return Admission(self._free_slots[-1], ck, sum(need), need)

    def admit(self, admission: Admission, prompt: tuple[int, ...], max_new: int) -> Sequence:
        slot = self._free_slots.pop()
        if slot != admission.slot:
            raise RuntimeError("admission planned for another slot")
        self.pool.reserve(admission.need)
        ck = admission.checkpoint
        start = len(ck.tokens) if ck else 0
        seq = Sequence(slot, prompt, max_new, start=start, reserved=list(admission.need), source=ck)
        if ck is not None:
            self.checkpoints.touch(ck)
            ck.users.add(slot)
            full, partial = divmod(seq.start, PAGE_CELLS)
            for logical in range(full):
                physical = ck.pages[logical]
                self.pool.incref(physical)
                seq.pages.append(physical)
                self._changes.append(PageChange(PageOp.ASSIGN, slot, logical, physical))
            if partial:
                physical = self._take(seq, full)
                seq.pages.append(physical)
                self._changes.append(PageChange(PageOp.COPY, slot, full, physical, ck.pages[full], partial))
        self.sequences[slot] = seq
        return seq

    def _take(self, seq: Sequence, logical: int) -> int:
        k = logical % self.pool.classes
        if seq.reserved[k] <= 0:
            raise RuntimeError(f"slot {seq.slot} grew past its reservation")
        seq.reserved[k] -= 1
        return self.pool.take(logical)

    def grow(self, slot: int, cells: int) -> None:
        """Give the slot pages covering its first `cells` cells, within its reservation."""
        seq = self.sequences[slot]
        cells = min(cells, len(seq.prompt) + seq.max_new)
        while seq.cells < cells:
            physical = self._take(seq, len(seq.pages))
            self._changes.append(PageChange(PageOp.ASSIGN, slot, len(seq.pages), physical))
            seq.pages.append(physical)

    def take_changes(self, limit: int) -> list[PageChange]:
        """Page changes for the next plan, in order, at most `limit`; the rest wait for the plan after."""
        out, self._changes = self._changes[:limit], self._changes[limit:]
        return out

    @property
    def pending_changes(self) -> int:
        return len(self._changes)

    def changes_pending_for(self, slot: int) -> bool:
        return any(c.slot == slot for c in self._changes)

    def submitted(self) -> None:
        """The plan holding the pending changes was submitted: copies from checkpoints are queued in the core."""
        if self._changes:
            return
        for seq in self.sequences.values():
            if seq.source is not None:
                seq.source.users.discard(seq.slot)
                seq.source = None

    def checkpoint(self, slot: int, position: int, kept: bool) -> Checkpoint | None:
        """Record a checkpoint of the slot's state after `position` prompt tokens. None if an equal one exists."""
        seq = self.sequences[slot]
        tokens = seq.prompt[:position]
        existing = self.checkpoints.find(tokens)
        if existing is not None:
            existing.kept = existing.kept or kept
            self.checkpoints.touch(existing)
            return None
        ck, evicted = self.checkpoints.add(tokens, tuple(seq.pages[: pages_for(position)]), kept)
        self.drops.extend(evicted)
        return ck

    def release(self, slot: int) -> None:
        seq = self.sequences.pop(slot)
        for physical in seq.pages:
            self.pool.decref(physical)
        self.pool.unreserve(seq.reserved)
        if seq.source is not None:
            seq.source.users.discard(slot)
        self._changes = [c for c in self._changes if c.slot != slot]
        self._free_slots.append(slot)
