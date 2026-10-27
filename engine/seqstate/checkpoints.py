"""The checkpoint pool: copies of a sequence's fixed state at a known prefix (DESIGN.md, section 4; ADR-0017).

A shared prefix can reuse its KV pages, but the DeltaNet state only through a checkpoint taken where the prefix
ends. A checkpoint here is the id of a state copy in the core, the tokens it was taken after, and the KV pages that
cover those tokens (one reference each). Prefixes are keyed by the rendered tokens, never by text (ADR-0017,
item 4).

A new checkpoint past the pool's size drops one that keeps each conversation's checkpoints spread back from its end,
the newest of each last (`spread_victim`, ADR-0051). Room is made least recently used first. A kept checkpoint (the
end of a long system prompt, ADR-0017 item 6) goes only when no other is left to evict.
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
    parent: Checkpoint | None = None
    """The longest other checkpoint whose tokens are a strict prefix of these."""


def extends(longer: tuple[int, ...], shorter: tuple[int, ...]) -> bool:
    """Whether `shorter` is a strict prefix of `longer`. The last token is checked first, which rejects most pairs."""
    n = len(shorter)
    return 0 < n < len(longer) and longer[n - 1] == shorter[-1] and longer[:n] == shorter


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
            victim = self.spread_victim(tokens)
            if victim is None:
                raise RuntimeError("every checkpoint is in use")
            evicted.append(victim.id)
            self.remove(victim.id)
        ck = Checkpoint(self._free.pop(), tokens, pages, kept)
        for page in pages:
            self._pages.incref(page)
        for other in self._by_id.values():
            if extends(tokens, other.tokens) and (ck.parent is None or len(other.tokens) > len(ck.parent.tokens)):
                ck.parent = other
            elif extends(other.tokens, tokens) and (other.parent is None or len(other.parent.tokens) < len(tokens)):
                other.parent = ck
        self.touch(ck)
        self._by_id[ck.id] = ck
        return ck, evicted

    def victim(self) -> Checkpoint | None:
        """The checkpoint to evict next: the least recently used not in use, kept ones last."""
        candidates = [ck for ck in self._by_id.values() if not ck.users and ck.ready]
        if not candidates:
            return None
        return min(candidates, key=lambda c: (c.kept, c.used))

    def spread_victim(self, incoming: tuple[int, ...] = ()) -> Checkpoint | None:
        """The checkpoint to evict for a new one of `incoming` tokens: of those a longer one extends, the new one
        counted, the one whose removal leaves the smallest gap (from the one before it, or position 0, to the nearest
        after it) for its distance from the longest after it. So the points left thin out with their distance from
        each conversation's end, and a client that rewrites its history d tokens back reads about d again. `victim`
        when there is none."""
        nearest: dict[int, int] = {}
        end: dict[int, int] = {}

        def after(parent: Checkpoint | None, n: int) -> None:
            if parent is not None:
                nearest[parent.id] = min(nearest.get(parent.id, n), n)
            while parent is not None:
                end[parent.id] = max(end.get(parent.id, n), n)
                parent = parent.parent

        for ck in self._by_id.values():
            after(ck.parent, len(ck.tokens))
        if incoming:
            prefixes = [ck for ck in self._by_id.values() if extends(incoming, ck.tokens)]
            after(max(prefixes, key=lambda c: len(c.tokens), default=None), len(incoming))
        best: Checkpoint | None = None
        best_key = (0.0, 0)
        for ck in self._by_id.values():
            if ck.id not in nearest or ck.kept or ck.users or not ck.ready:
                continue
            base = len(ck.parent.tokens) if ck.parent is not None else 0
            key = ((nearest[ck.id] - base) / (end[ck.id] - base), ck.used)
            if best is None or key < best_key:
                best, best_key = ck, key
        return best if best is not None else self.victim()

    def remove(self, checkpoint_id: int) -> None:
        ck = self._by_id.pop(checkpoint_id)
        for other in self._by_id.values():
            if other.parent is ck:
                other.parent = ck.parent
        for page in ck.pages:
            self._pages.decref(page)
        self._free.append(ck.id)
