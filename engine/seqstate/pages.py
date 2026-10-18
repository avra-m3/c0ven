"""The KV page pool: physical page ids, reference counts, free lists and reservations (DESIGN.md, section 4).

Only integers live here. The bytes are the core's; the page ids are the same on both cards, so one table serves
both. A page is shared by reference count: a checkpoint holds the pages of the prefix it was taken at, and every
sequence resumed from it holds them too.

The pages fall in classes, one per card: the indexer's keys are split by position, so a sequence's logical page i
must be a page of class i % classes (`include/engine_api.h`). Each class has its own free list and reservations.

A sequence reserves, at admission, every page it can ever need (prompt plus `max_new`), and takes them one by one
as it grows. So a running sequence never finds the pool empty.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence


class PoolExhausted(RuntimeError):
    pass


type Counts = int | Sequence[int]
"""Pages per class; an int for a pool of one class."""


class PagePool:
    def __init__(self, pages: int, classes: int = 1) -> None:
        if pages <= 0:
            raise ValueError("a pool needs at least one page")
        if classes <= 0 or pages % classes:
            raise ValueError(f"{pages} pages do not fall evenly in {classes} classes")
        self.classes = classes
        self._free: list[deque[int]] = [deque(range(k, pages, classes)) for k in range(classes)]
        self._refs = [0] * pages
        self._reserved = [0] * classes
        self.size = pages

    def _counts(self, n: Counts) -> list[int]:
        if isinstance(n, int):
            if self.classes != 1:
                raise ValueError("a pool of several classes counts pages per class")
            n = (n,)
        if len(n) != self.classes or any(k < 0 for k in n):
            raise ValueError(f"bad page counts {tuple(n)}")
        return list(n)

    def split(self, first: int, end: int) -> tuple[int, ...]:
        """How many of the logical pages first .. end - 1 fall in each class."""
        return tuple(len(range(first + (k - first) % self.classes, end, self.classes)) for k in range(self.classes))

    @property
    def free(self) -> int:
        """Pages on the free lists, reserved or not."""
        return sum(len(f) for f in self._free)

    @property
    def available(self) -> int:
        """Pages that can still be reserved, summed over the classes."""
        return self.free - self.reserved

    @property
    def reserved(self) -> int:
        return sum(self._reserved)

    def can_reserve(self, n: Counts) -> bool:
        return all(k <= len(f) - r for k, f, r in zip(self._counts(n), self._free, self._reserved, strict=True))

    def refs(self, page: int) -> int:
        return self._refs[page]

    def reserve(self, n: Counts) -> None:
        counts = self._counts(n)
        if not self.can_reserve(counts):
            raise PoolExhausted(f"{tuple(counts)} pages wanted, {self.available} available")
        for k, c in enumerate(counts):
            self._reserved[k] += c

    def unreserve(self, n: Counts) -> None:
        counts = self._counts(n)
        if any(c > r for c, r in zip(counts, self._reserved, strict=True)):
            raise ValueError(f"unreserve {tuple(counts)} of {tuple(self._reserved)}")
        for k, c in enumerate(counts):
            self._reserved[k] -= c

    def take(self, logical: int = 0) -> int:
        """A page for logical page `logical`, from the caller's reservation, with one reference."""
        k = logical % self.classes
        if self._reserved[k] <= 0:
            raise PoolExhausted("take without a reservation")
        self._reserved[k] -= 1
        page = self._free[k].popleft()
        self._refs[page] = 1
        return page

    def incref(self, page: int) -> None:
        if self._refs[page] <= 0:
            raise ValueError(f"page {page} is free")
        self._refs[page] += 1

    def decref(self, page: int) -> None:
        if self._refs[page] <= 0:
            raise ValueError(f"page {page} is already free")
        self._refs[page] -= 1
        if self._refs[page] == 0:
            self._free[page % self.classes].append(page)
