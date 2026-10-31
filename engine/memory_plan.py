"""The memory plan: what each card and the host hold, computed before `open` (DESIGN.md, section 6).

Everything is planned, then allocated, at load. Python computes the plan from the tensor table and the options and
checks it against the host's memory (engine/host.py); `open` allocates it, and nothing is allocated on the token
path. The method is MEMORY_BUDGET.md's for the tensor split: per card, the fixed rows, then a reserve per sequence,
then experts in what is left; in the host, the arena of the experts the cards do not hold and the streamed KV.

The model supplies its numbers as a `ModelMemory` (models/<arch>/memory.py).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from engine.api.interface import PAGE_CELLS

GIB = 2**30


class PlanError(ValueError):
    """The options do not fit."""


@dataclass(frozen=True, slots=True)
class ModelMemory:
    """A model's sizes on each of `cards` cards. Every "per card" figure is one card's share."""

    cards: int
    weights_per_card: int
    """Every tensor outside the expert cache: dense weights, embedding, head, PLE."""
    draft_per_card: int
    layers: int
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
    ngram_bytes: int
    """The n-gram table, which stays in the file unless the plan holds it in RAM."""
    reference_vram: int
    """One test-machine card's memory, bytes: the plan's card when none is read (the mock and the reference)."""


@dataclass(frozen=True, slots=True)
class PlanOptions:
    slots: int = 1
    context: int = 262_144
    cards: int = 2
    """Cards the engine runs on: the tensor split's two, or one holding every weight whole."""
    usable_vram: int | None = None
    """The smallest card's memory, bytes, as Level Zero reports it; None: the model's reference card."""
    draft: bool = True
    """The draft layer is loaded, and takes its share of each card."""
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
    host_working: int = 3 * GIB
    """*Estimate*: the engine process's host memory outside the pinned total: staging and step buffers, the n-gram
    cache, the front end."""
    host_memory: int | None = None
    """The most host memory the engine can take, read from the host (engine/host.py); None: no limit known."""
    ram_max: int | None = None
    """`--ram-max`: a limit the user sets on the host memory the plan may take; None: none."""
    ngram_in_ram: bool = False
    """The whole n-gram table in RAM, counted in the host total."""


@dataclass(frozen=True, slots=True)
class MemoryPlan:
    options: PlanOptions
    cards: int
    usable_vram: int
    layers: int
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
    ngram_bytes: int

    @property
    def pinned_bytes(self) -> int:
        """Everything `open` pins in host memory, which a wake pins again after a sleep but for the arena."""
        return self.arena_bytes + self.streamed_kv_bytes + self.checkpoint_bytes

    @property
    def experts_per_layer(self) -> int:
        """The experts each layer caches, as the core reports its own count."""
        return self.experts_cached // self.layers

    @property
    def host_bytes(self) -> int:
        """The host memory the engine takes: the pinned total, the working estimate, and the n-gram table when it is
        in RAM."""
        return self.pinned_bytes + self.options.host_working + (self.ngram_bytes if self.options.ngram_in_ram else 0)

    def rows(self) -> list[tuple[str, str]]:
        def gib(n: int) -> str:
            return f"{n:,} B ({n / GIB:.2f} GiB)"

        o = self.options
        return [
            ("sequences x context", f"{o.slots} x {o.context:,}"),
            ("cards", str(self.cards)),
            ("usable VRAM per card", gib(self.usable_vram)),
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
            ("n-gram table", f"{gib(self.ngram_bytes)}, {'in RAM' if o.ngram_in_ram else 'read from the file'}"),
            ("host memory with working set (estimate)", gib(self.host_bytes)),
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
    """The plan for the options. Raises PlanError, with options that fit where there are some, when the cards or the
    host's memory are too small."""
    try:
        p = _plan(m, o)
    except PlanError as exc:
        if not exc.args[0].startswith("the cards are"):
            raise
        raise PlanError(f"{exc} {_what_fits(m, o, host_limit(o))}") from None
    limit = host_limit(o)
    if limit is not None and p.host_bytes > limit:
        raise PlanError(
            f"the engine needs {p.host_bytes / GIB:.1f} GiB of host memory ({p.pinned_bytes / GIB:.1f} GiB pinned: "
            f"arena {p.arena_bytes / GIB:.1f}, streamed KV {p.streamed_kv_bytes / GIB:.1f}, checkpoints "
            f"{p.checkpoint_bytes / GIB:.1f}; working set {o.host_working / GIB:.1f}, an estimate), over the "
            f"{limit / GIB:.1f} GiB {_limit_name(o)}. {_what_fits(m, o, limit)}"
        )
    return p


NGRAM_MARGIN_SHARE = 0.25
"""The share of the host's memory, at least 8 GiB, left free after the n-gram table moves into RAM: the table is the
first thing the host would want back."""


def ngram_fits(p: MemoryPlan, mem_total: int | None) -> bool:
    """Whether the host has room for the n-gram table in RAM beside the plan (the plan's own table not counted)."""
    limit = host_limit(p.options)
    if limit is None or mem_total is None:
        return False
    total = p.host_bytes - (p.ngram_bytes if p.options.ngram_in_ram else 0)
    return limit - total - p.ngram_bytes >= max(8 * GIB, NGRAM_MARGIN_SHARE * mem_total)


def host_limit(o: PlanOptions) -> int | None:
    known = [v for v in (o.host_memory, o.ram_max) if v is not None]
    return min(known) if known else None


def _limit_name(o: PlanOptions) -> str:
    if o.ram_max is not None and (o.host_memory is None or o.ram_max <= o.host_memory):
        return "--ram-max allows"
    return "the host can give (MemAvailable, or the cgroup's memory.max)"


def _what_fits(m: ModelMemory, o: PlanOptions, limit: int | None) -> str:
    """The largest options that fit the cards and the host limit: fewer slots first, then a shorter context, no
    checkpoints, no draft layer; or what to change when none does. The arena holds every expert the cards leave out,
    so a smaller model file is the remaining way."""
    fits: list[str] = []
    for slots in range(o.slots - 1, 0, -1):
        if _fits(m, replace(o, slots=slots, kv_tokens=0 if o.kv_tokens < o.context else o.kv_tokens), limit):
            fits.append(f"--slots {slots}")
            break
    context = o.context
    while context > 4096:
        context //= 2
        if _fits(m, replace(o, context=context, kv_tokens=0), limit):
            fits.append(f"--context {context} (with --slots {o.slots})")
            break
    if o.checkpoints and _fits(m, replace(o, checkpoints=0), limit):
        fits.append("--checkpoints 0")
    if o.draft and _fits(m, replace(o, draft=False), limit):
        fits.append("--no-draft")
    if not fits:
        return "No slot count or context fits: free host memory, or use a smaller model file."
    return "Options that fit: " + "; ".join(fits) + "; or a smaller model file."


def _fits(m: ModelMemory, o: PlanOptions, limit: int | None) -> bool:
    try:
        return limit is None or _plan(m, o).host_bytes <= limit
    except PlanError:
        return False


def _plan(m: ModelMemory, o: PlanOptions) -> MemoryPlan:
    if o.slots < 1:
        raise PlanError("at least one slot")
    if not 0 < o.context <= m.max_context:
        raise PlanError(f"context {o.context} outside 1..{m.max_context}")
    vram = o.usable_vram if o.usable_vram is not None else m.reference_vram
    draft = m.draft_per_card if o.draft else 0
    fixed = m.weights_per_card + draft + o.working_per_card
    per_sequence = reserve_per_sequence(m, o.context)
    reserve = o.slots * per_sequence
    free = vram - fixed - reserve
    if free < 0:
        raise PlanError(
            f"the cards are {-free:,} B short before any expert: each has {vram / GIB:.2f} GiB, and the weights take "
            f"{m.weights_per_card / GIB:.2f}, the draft layer {draft / GIB:.2f}, working memory "
            f"{o.working_per_card / GIB:.2f} and {o.slots} x {o.context:,} positions {reserve / GIB:.2f}."
        )
    share = -(-m.expert_bytes // m.cards)
    cached = min(m.experts, free // share)
    arena = (m.experts - cached) * m.expert_bytes
    pages_per_slot = -(-o.context // PAGE_CELLS)
    if 0 < o.kv_tokens < o.context:
        raise PlanError(f"a KV pool of {o.kv_tokens:,} tokens holds no sequence of {o.context:,}")
    pages = -(-o.kv_tokens // PAGE_CELLS) if o.kv_tokens else o.slots * pages_per_slot + o.cached_prefix_pages
    # The host holds every page's K and V (the core's kv_host), and each slot's draft ring.
    streamed = pages * PAGE_CELLS * m.kv_cell_host + o.slots * m.draft_kv_host
    checkpoints = o.checkpoints * m.checkpoint_bytes
    return MemoryPlan(
        options=o,
        cards=m.cards,
        usable_vram=vram,
        layers=m.layers,
        weights_per_card=m.weights_per_card,
        draft_per_card=draft,
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
        ngram_bytes=m.ngram_bytes,
    )
