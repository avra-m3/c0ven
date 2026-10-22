"""The checkpoint pool: copies of a sequence's fixed state at a known prefix (DESIGN.md, section 4; ADR-0017).

A shared prefix can reuse its KV pages, but the DeltaNet state only through a checkpoint taken where the prefix
ends. A checkpoint here is the id of a state copy in the core, the tokens it was taken after, and the KV pages that
cover those tokens (one reference each). Prefixes are keyed by the rendered tokens, never by text (ADR-0017,
item 4).

Eviction is least recently used. A kept checkpoint (the end of a long system prompt, ADR-0017 item 6) goes only
when no other is left to evict.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from engine.seqstate.pages import PagePool


@dataclass(slots=True)
class Checkpoint:
    id: int
    tokens: tuple[int, ...]
    pages: tuple[int, ...]
    kept: bool
    ready: bool = False
    """The core has answered CHECKPOINT_DONE."""
    used: int = 0
    users: set[int] = field(default_factory=set[int])
    """Slots being restored from it, which it must outlive until their restore is queued."""


class CheckpointPool:
    def __init__(self, size: int, pages: PagePool, max_kept: int = 2) -> None:
        self._free = list(range(size - 1, -1, -1))
        self._pages = pages
        self._by_id: dict[int, Checkpoint] = {}
        self._clock = itertools.count(1)
        self.max_kept = max_kept
        self.size = size

    def __len__(self) -> int:
        return len(self._by_id)

    def __iter__(self):  # noqa: ANN204
        return iter(self._by_id.values())

    def get(self, checkpoint_id: int) -> Checkpoint | None:
        return self._by_id.get(checkpoint_id)

    def longest_prefix(self, prompt: tuple[int, ...]) -> Checkpoint | None:
        """The ready checkpoint with the longest tokens that are a strict prefix of `prompt`."""
        best: Checkpoint | None = None
        for ck in self._by_id.values():
            n = len(ck.tokens)
            if not ck.ready or n >= len(prompt) or (best is not None and n <= len(best.tokens)):
                continue
            if prompt[:n] == ck.tokens:
                best = ck
        return best

    def find(self, tokens: tuple[int, ...]) -> Checkpoint | None:
        for ck in self._by_id.values():
            if ck.tokens == tokens:
                return ck
        return None

    def touch(self, ck: Checkpoint) -> None:
        ck.used = next(self._clock)

    def add(self, tokens: tuple[int, ...], pages: tuple[int, ...], kept: bool) -> tuple[Checkpoint, list[int]]:
        """A new checkpoint holding a reference on each of `pages`, and the ids of any evicted to make room."""
        evicted: list[int] = []
        if kept and sum(ck.kept for ck in self._by_id.values()) >= self.max_kept:
            kept_now = [ck for ck in self._by_id.values() if ck.kept]
            min(kept_now, key=lambda c: c.used).kept = False
        if not self._free:
            victim = self.victim()
            if victim is None:
                raise RuntimeError("every checkpoint is in use")
            evicted.append(victim.id)
            self.remove(victim.id)
        ck = Checkpoint(self._free.pop(), tokens, pages, kept)
        for page in pages:
            self._pages.incref(page)
        self.touch(ck)
        self._by_id[ck.id] = ck
        return ck, evicted

    def victim(self) -> Checkpoint | None:
        """The checkpoint to evict next: the least recently used not in use, kept ones last."""
        candidates = [ck for ck in self._by_id.values() if not ck.users and ck.ready]
        if not candidates:
            return None
        return min(candidates, key=lambda c: (c.kept, c.used))

    def remove(self, checkpoint_id: int) -> None:
        ck = self._by_id.pop(checkpoint_id)
        for page in ck.pages:
            self._pages.decref(page)
        self._free.append(ck.id)
