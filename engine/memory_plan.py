"""The memory plan: what each card and the host hold, computed before `open` (DESIGN.md, section 6).

Everything is planned, then allocated, at load. Python computes the plan from the tensor table and the options and
checks it against the host's limits; `open` allocates it, and nothing is allocated on the token path. The method is
MEMORY_BUDGET.md's for the tensor split: per card, the fixed rows, then a reserve per sequence, then experts in
what is left; in the host, the arena of the experts the cards do not hold and the streamed KV.

The model supplies its numbers as a `ModelMemory` (models/<arch>/memory.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.api.interface import PAGE_CELLS

GIB = 2**30


class PlanError(ValueError):
    """The options do not fit."""


@dataclass(frozen=True, slots=True)
class ModelMemory:
    """A model's sizes, per card for the tensor split. Every "per card" figure is one card's half or whole share."""

    weights_per_card: int
    """Every tensor outside the expert cache: dense weights, embedding, head, PLE."""
    draft_per_card: int
    experts: int
    expert_bytes: int
    """The mean size of one expert, whole."""
    kv_cell_per_card: int
    """K and V per token, one card's heads."""
    indexer_cell_per_card: int
    """The indexer's keys per token, one card's half of the blocks."""
    state_per_card: int
    draft_kv_per_card: int
    resident_cells: int
    """K and V cells kept in VRAM per sequence; the rest streams from host memory."""
    kv_cell_host: int
    """K and V per token in host memory, both cards' heads: the whole authoritative copy."""
    draft_kv_host: int
    checkpoint_bytes: int
    """A checkpoint: the running state only; the KV of its positions stays in its pages (ADR-0033)."""
    max_context: int


@dataclass(frozen=True, slots=True)
class PlanOptions:
    slots: int = 1
    context: int = 262_144
    usable_vram: int = 24_385_683_456
    working_per_card: int = 1_600_663_296
    checkpoints: int = 8
    kv_tokens: int = 0
    """The KV pool the slots share, in tokens; 0: every slot's own context, plus cached_prefix_pages. It is
    independent of the slot count, and at least one context: a request waits until the pool can reserve its pages
    (seqstate/pages.py)."""
    cached_prefix_pages: int = 0
    """Pages beyond the slots' own, for prefixes that checkpoints keep after their requests end. Every page's indexer
    keys stay in VRAM on both cards (ADR-0033), so each costs experts; with none, prefixes share the pages of the
    slots that are not full."""
    pinned_clean: int = 12 * GIB
    pinned_limit: int = 25 * GIB


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    options: PlanOptions
    weights_per_card: int
    draft_per_card: int
    working_per_card: int
    reserve_per_sequence: int
    reserve_per_card: int
    free_per_card: int
    experts_cached: int
    experts_total: int
    arena_bytes: int
    streamed_kv_bytes: int
    checkpoint_bytes: int
    pages: int
    kv_resident_cells: int
    warnings: tuple[str, ...] = field(default=())

    @property
    def pinned_bytes(self) -> int:
        """Everything `open` pins in host memory, which a wake pins again after a sleep but for the arena."""
        return self.arena_bytes + self.streamed_kv_bytes + self.checkpoint_bytes

    def rows(self) -> list[tuple[str, str]]:
        def gib(n: int) -> str:
            return f"{n:,} B ({n / GIB:.2f} GiB)"

        o = self.options
        return [
            ("sequences x context", f"{o.slots} x {o.context:,}"),
            ("usable VRAM per card", gib(o.usable_vram)),
            ("weights per card", gib(self.weights_per_card)),
            ("draft layer per card (estimate)", gib(self.draft_per_card)),
            ("working memory per card (estimate)", gib(self.working_per_card)),
            ("reserve per sequence per card", gib(self.reserve_per_sequence)),
            ("free for experts per card", gib(self.free_per_card)),
            ("experts cached", f"{self.experts_cached:,} of {self.experts_total:,}"),
            ("host arena", gib(self.arena_bytes)),
            ("streamed KV", gib(self.streamed_kv_bytes)),
            ("checkpoints", f"{o.checkpoints} x {self.checkpoint_bytes // o.checkpoints if o.checkpoints else 0:,} B"),
            ("pinned host memory", gib(self.pinned_bytes)),
            ("KV pages", f"{self.pages:,} of {PAGE_CELLS} cells"),
        ]


def reserve_per_sequence(m: ModelMemory, context: int) -> int:
    """One sequence's share of one card: resident K and V, the indexer's keys, the state, the draft ring."""
    return (
        m.kv_cell_per_card * min(context, m.resident_cells)
        + m.indexer_cell_per_card * context
        + m.state_per_card
        + m.draft_kv_per_card
    )


def plan(m: ModelMemory, o: PlanOptions) -> MemoryPlan:
    if o.slots < 1:
        raise PlanError("at least one slot")
    if not 0 < o.context <= m.max_context:
        raise PlanError(f"context {o.context} outside 1..{m.max_context}")
    fixed = m.weights_per_card + m.draft_per_card + o.working_per_card
    per_sequence = reserve_per_sequence(m, o.context)
    reserve = o.slots * per_sequence
    free = o.usable_vram - fixed - reserve
    if free < 0:
        raise PlanError(f"the cards are {-free:,} B short before any expert")
    half = -(-m.expert_bytes // 2)
    cached = min(m.experts, free // half)
    arena = (m.experts - cached) * m.expert_bytes
    streamed = o.slots * (m.kv_cell_host * o.context + m.draft_kv_host)
    pages_per_slot = -(-o.context // PAGE_CELLS)
    if 0 < o.kv_tokens < o.context:
        raise PlanError(f"a KV pool of {o.kv_tokens:,} tokens holds no sequence of {o.context:,}")
    pages = -(-o.kv_tokens // PAGE_CELLS) if o.kv_tokens else o.slots * pages_per_slot + o.cached_prefix_pages
    checkpoints = o.checkpoints * m.checkpoint_bytes
    pinned = arena + streamed + checkpoints
    warnings: list[str] = []
    # Both limits are warnings: what the host gives depends on how fragmented its memory is (MEMORY_BUDGET.md, "RAM
    # for the tensor split"). `open` is the test that counts.
    if pinned > o.pinned_limit:
        warnings.append(
            f"pinned host memory {pinned / GIB:.1f} GiB is over {o.pinned_limit / GIB:.1f} GiB, where the test host "
            "began to swap"
        )
    elif pinned > o.pinned_clean:
        warnings.append(
            f"pinned host memory {pinned / GIB:.1f} GiB is over {o.pinned_clean / GIB:.1f} GiB: the host will compact "
            "memory and reclaim its page cache while the engine loads"
        )
    return MemoryPlan(
        options=o,
        weights_per_card=m.weights_per_card,
        draft_per_card=m.draft_per_card,
        working_per_card=o.working_per_card,
        reserve_per_sequence=per_sequence,
        reserve_per_card=reserve,
        free_per_card=free,
        experts_cached=cached,
        experts_total=m.experts,
        arena_bytes=arena,
        streamed_kv_bytes=streamed,
        checkpoint_bytes=checkpoints,
        pages=pages,
        kv_resident_cells=min(o.context, m.resident_cells),
        warnings=tuple(warnings),
    )
