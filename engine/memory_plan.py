"""The memory plan: what each card and the host hold, computed before `open` (DESIGN.md, section 6).

Python computes the plan from the tensor table and the options, by MEMORY_BUDGET.md's method, and checks it against the
host's memory (engine/host.py). `open` allocates it, and nothing is allocated on the token path. The model supplies its
numbers as a `ModelMemory` (models/<arch>/memory.py).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

from engine.api.interface import PAGE_CELLS

GIB = 2**30

SSD_TIER_HEADROOM = 24 * GIB
ARENA_PIECE = GIB
"""The core takes its arena in pieces of this many bytes (the core's kArenaPiece)."""
"""What a plan with an SSD tier leaves free on the host, bytes: the 22 GiB default of `tools/check.py --mem-floor` and
2 GiB, so a timed run does not stop at the floor."""


class PlanError(ValueError):
    """The options do not fit."""


@dataclass(frozen=True, slots=True)
class ModelMemory:
    """A model's sizes on each of `cards` cards. Every "per card" figure is one card's share."""

    cards: int
    weights_per_card: int
    """Every tensor outside the expert cache: dense weights, embedding, head, PLE."""
    draft_per_card: int
    working_per_card: int
    """*Estimate*: each card's memory outside the weights, the draft layer, the sequences' reserve and the expert
    cache that no option changes. PlanOptions.working_per_card, the core's figure, replaces this and `working_of`."""
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
    working_of: Callable[[PlanOptions, int], int] | None = None
    """*Estimate*: what the options add to `working_per_card`, given them and the KV pool in tokens; None: nothing."""
    card_weights: tuple[int, ...] = ()
    """Each card's weights, bytes, where the cards differ, in place of `weights_per_card`; empty: all hold that."""
    layer_expert_bytes: tuple[int, ...] = ()
    """The mean size of one expert, whole, in each layer, bytes; empty: every layer's is `expert_bytes`."""
    ssd_tier: bool = False
    """The experts VRAM and the arena do not hold stay on the SSD. The arena takes what the host can spare and may be
    empty; without this it holds every expert VRAM does not."""
    engram_cache: int = 0
    """The Engram row cache and its tags, host bytes."""
    ssd_staging_ring: int = 0
    """The ring decode's SSD reads fill, host bytes."""
    prompt_staging_per_card: int = 0
    """Each card's prompt staging slots, bytes, taken before the experts."""
    staging_per_uncached: int = 0
    """Each card's prompt staging for one expert a layer it does not cache, bytes. The core takes it from what the
    experts have, so fewer cached experts cost more of it; `free_per_card` stays the memory before it."""
    prompt_host_ring: int = 0
    """The ring prompts stage SSD-tier experts through, host bytes, taken before the arena."""
    routes_per_row: int = 0
    """The experts one row routes to in each layer, for the SSD reads a token; 0: not given."""
    hit_curve: tuple[tuple[float, float], ...] = ()
    """A routing profile's hit curve: (share of a layer's experts, hottest first; share of routings they serve), both
    rising, from 0 to 1 between the points given; empty: no profile."""

    def __post_init__(self) -> None:
        if self.card_weights and len(self.card_weights) != self.cards:
            raise ValueError(f"{len(self.card_weights)} card weights for {self.cards} cards")
        if self.layer_expert_bytes and len(self.layer_expert_bytes) != self.layers:
            raise ValueError(f"{len(self.layer_expert_bytes)} expert sizes for {self.layers} layers")
        if (self.layer_expert_bytes or self.ssd_tier) and not 0 < self.layers <= self.experts:
            raise ValueError(f"{self.experts} experts for {self.layers} layers: each layer needs at least one")

    @property
    def weights(self) -> tuple[int, ...]:
        """Each card's weights."""
        return self.card_weights or (self.weights_per_card,) * self.cards


@dataclass(frozen=True, slots=True)
class PlanOptions:
    slots: int = 1
    context: int = 0
    """Each slot's context, in tokens; 0: the model's whole context (ModelMemory.max_context), which `plan` puts in
    its MemoryPlan.options."""
    cards: int = 2
    """Cards the engine runs on: the tensor split's two, or one holding every weight whole."""
    usable_vram: int | None = None
    """The smallest card's memory, bytes, as Level Zero reports it, or the mock card's (api/mock.py) where no card is
    read. None: not known, which `plan` refuses."""
    draft: bool = True
    """The draft layer is loaded, and takes its share of each card."""
    working_per_card: int | None = None
    """Each card's working memory, as ModelMemory.working_per_card: the core's figure from an earlier load when
    `ENGINE_CACHE_DIR` keeps one (working_file); None: the model's estimate."""
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
    hugetlb_pool: int | None = None
    """`--hugepages`: the host's free reserved huge pages, bytes. The arena's pieces take them first, and what they
    hold is not counted against host_memory, which leaves them out. None: the arena is all in host memory."""


@dataclass(frozen=True, slots=True)
class Tiers:
    """Where a model with an SSD tier holds its experts, as counts per layer. An expert is in one tier only; filling
    goes hottest first into VRAM, the next into the arena, and the SSD holds the rest."""

    layer_expert_bytes: tuple[int, ...]
    """The mean size of one expert, whole, in each layer, bytes."""
    vram: tuple[int, ...]
    """Experts each layer holds in VRAM, across the cards."""
    arena: tuple[int, ...]
    """Experts each layer holds in the host arena."""
    ssd: tuple[int, ...]
    """Experts each layer leaves on the SSD."""
    routes_per_row: int = 0
    """As ModelMemory.routes_per_row."""
    hit_curve: tuple[tuple[float, float], ...] = ()
    """As ModelMemory.hit_curve."""

    def bytes_of(self, experts: tuple[int, ...]) -> int:
        """The bytes of one tier's experts, whole."""
        return sum(n * b for n, b in zip(experts, self.layer_expert_bytes, strict=True))

    def with_core(self, experts_per_layer: int) -> Tiers:
        """The tiers with VRAM at the core's count, which is one figure across the layers: each layer's count scaled by
        it. The arena keeps what it holds, and the SSD takes the difference."""
        planned = sum(self.vram)
        layers = len(self.vram)
        totals = [v + a + s for v, a, s in zip(self.vram, self.arena, self.ssd, strict=True)]
        vram = tuple(
            min(t, round(v * experts_per_layer * layers / planned) if planned else experts_per_layer)
            for v, t in zip(self.vram, totals, strict=True)
        )
        arena = tuple(min(a, t - v) for a, t, v in zip(self.arena, totals, vram, strict=True))
        ssd = tuple(t - v - a for t, v, a in zip(totals, vram, arena, strict=True))
        return replace(self, vram=vram, arena=arena, ssd=ssd)

    def _served(self, held: float) -> float:
        """The hit curve at a share of a layer's experts."""
        points = [(0.0, 0.0), *self.hit_curve, (1.0, 1.0)]
        for (x0, y0), (x1, y1) in pairwise(points):
            if held <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (held - x0) / (x1 - x0)
        return 1.0

    def routing_shares(self) -> tuple[float, float, float] | None:
        """The share of routings VRAM, the arena and the SSD each serve, every layer routing alike; None: no profile."""
        if not self.hit_curve:
            return None
        vram = arena = 0.0
        for v, a, s in zip(self.vram, self.arena, self.ssd, strict=True):
            total = v + a + s
            vram += self._served(v / total)
            arena += self._served((v + a) / total) - self._served(v / total)
        layers = len(self.vram)
        return vram / layers, arena / layers, 1.0 - (vram + arena) / layers

    def ssd_per_token(self) -> tuple[float, float]:
        """The SSD reads and bytes a token expects at one row, from the hit curve or, without one, at uniform routing,
        the worst case."""
        reads = size = 0.0
        for v, a, s, b in zip(self.vram, self.arena, self.ssd, self.layer_expert_bytes, strict=True):
            total = v + a + s
            share = 1.0 - self._served((v + a) / total) if self.hit_curve else s / total
            reads += self.routes_per_row * share
            size += self.routes_per_row * share * b
        return reads, size


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
    card_weights: tuple[int, ...] = ()
    """Each card's weights; `weights_per_card` is the largest."""
    card_free: tuple[int, ...] = ()
    """Each card's memory free for experts; `free_per_card` is the smallest."""
    card_layer_experts: tuple[tuple[int, ...], ...] = ()
    """Per card, the experts each layer's bytes hold, when the model gives per-layer sizes. Each card holds its share of
    every cached expert, so the plan caches what the card with the least free holds. Empty: `experts_per_layer` is the
    one count."""
    tiers: Tiers | None = None
    """None: the model has no SSD tier."""
    engram_cache_bytes: int = 0
    """As ModelMemory.engram_cache, host bytes."""
    ssd_staging_bytes: int = 0
    """As ModelMemory.ssd_staging_ring, host bytes."""
    prompt_host_ring_bytes: int = 0
    """As ModelMemory.prompt_host_ring, host bytes."""
    prompt_staging_per_card: int = 0
    """As ModelMemory.prompt_staging_per_card, bytes on each card."""

    @property
    def pinned_bytes(self) -> int:
        """Everything `open` pins in host memory, which a wake pins again after a sleep but for the arena."""
        return self.arena_bytes + self.streamed_kv_bytes + self.checkpoint_bytes

    @property
    def experts_per_layer(self) -> int:
        """The experts each layer caches, as the core reports its own count."""
        return self.experts_cached // self.layers

    @property
    def arena_reserved_bytes(self) -> int:
        """The part of the arena in reserved huge pages (PlanOptions.hugetlb_pool): the core's whole pieces that fit
        in them."""
        if self.options.hugetlb_pool is None:
            return 0
        return min(self.arena_bytes, self.options.hugetlb_pool // ARENA_PIECE * ARENA_PIECE)

    @property
    def host_bytes(self) -> int:
        """The host memory the engine takes: the pinned total but for the arena's part in reserved huge pages, the
        working estimate, and the n-gram table when it is in RAM."""
        return (
            self.pinned_bytes
            - self.arena_reserved_bytes
            + self.options.host_working
            + (self.ngram_bytes if self.options.ngram_in_ram else 0)
            + self.host_caches_bytes
        )

    @property
    def host_caches_bytes(self) -> int:
        """The Engram cache and the staging rings, which a model without them leaves at 0."""
        return self.engram_cache_bytes + self.ssd_staging_bytes + self.prompt_host_ring_bytes

    def working_from(self, expert_budget: int) -> int:
        """The working memory per card that the core's load shows: the plan's figure, and the part of the plan's free
        memory the core did not have for experts. `expert_budget` is the core's, its smallest card's
        (EngineFacts.expert_budget)."""
        return self.working_per_card + self.free_per_card - expert_budget

    def rows(self) -> list[tuple[str, str]]:
        def gib(n: int) -> str:
            return f"{n:,} B ({n / GIB:.2f} GiB)"

        def range_of(counts: tuple[int, ...]) -> str:
            return f"{min(counts):,}-{max(counts):,}" if min(counts) != max(counts) else f"{min(counts):,}"

        o = self.options
        rows = [
            ("sequences x context", f"{o.slots} x {o.context:,}"),
            ("cards", str(self.cards)),
            ("usable VRAM per card", gib(self.usable_vram)),
            ("weights per card", gib(self.weights_per_card)),
            ("draft layer per card", gib(self.draft_per_card)),
            ("working memory per card (estimate)", gib(self.working_per_card)),
            ("reserve per sequence per card", gib(self.reserve_per_sequence)),
            ("free for experts per card", gib(self.free_per_card)),
            ("experts cached", f"{self.experts_cached:,} of {self.experts_total:,}"),
            ("host arena", gib(self.arena_bytes)),
            *(
                [("host arena in reserved huge pages", gib(self.arena_reserved_bytes))]
                if o.hugetlb_pool is not None
                else []
            ),
            ("streamed KV", gib(self.streamed_kv_bytes)),
            ("checkpoints", f"{o.checkpoints} x {self.checkpoint_bytes // o.checkpoints if o.checkpoints else 0:,} B"),
            ("pinned host memory", gib(self.pinned_bytes)),
            ("n-gram table", f"{gib(self.ngram_bytes)}, {'in RAM' if o.ngram_in_ram else 'read from the file'}"),
            ("host memory with working set (estimate)", gib(self.host_bytes)),
            ("KV pages", f"{self.pages:,} of {PAGE_CELLS} cells"),
        ]
        if len(set(self.card_weights)) > 1:
            rows += [(f"weights on card {i}", gib(w)) for i, w in enumerate(self.card_weights)]
        rows += [(f"experts a layer on card {i}", range_of(c)) for i, c in enumerate(self.card_layer_experts)]
        if (t := self.tiers) is not None:
            rows += [
                (f"{name} tier, experts a layer", f"{range_of(n)}, {gib(t.bytes_of(n))}")
                for name, n in (("VRAM", t.vram), ("arena", t.arena), ("SSD", t.ssd))
            ]
            rows += [
                ("Engram row cache", gib(self.engram_cache_bytes)),
                ("SSD staging ring", gib(self.ssd_staging_bytes)),
                ("prompt staging per card", gib(self.prompt_staging_per_card)),
                ("prompt host ring", gib(self.prompt_host_ring_bytes)),
                ("host headroom", gib(SSD_TIER_HEADROOM)),
            ]
        return rows


def working_file(cache: Path, model: Path, o: PlanOptions, more: str) -> Path:
    """Where the core's working memory per card is kept between starts, in `ENGINE_CACHE_DIR`: a file for
    each model file, engine version, and set of options that changes what the core allocates before its experts.
    `more` names the rest of those options (the draft layer, the vision encoder, ...)."""
    st = model.stat()
    version = (Path(__file__).resolve().parents[1] / "VERSION").read_text().strip()
    key = "\n".join(
        (
            f"{model.resolve()} {st.st_size} {st.st_mtime_ns}",
            version,
            f"cards={o.cards} vram={o.usable_vram} slots={o.slots} context={o.context} kv_tokens={o.kv_tokens} "
            f"prefix_pages={o.cached_prefix_pages}",
            more,
        )
    )
    return cache / f"working-{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"


def read_working(file: Path) -> int | None:
    """The working memory per card kept in `file`, or None if there is none to use."""
    try:
        n = json.loads(file.read_text())["working_per_card"]
    except OSError, ValueError, KeyError, TypeError:
        return None
    return n if isinstance(n, int) and n > 0 else None


def write_working(file: Path, n: int) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"working_per_card": n}) + "\n")
    tmp.replace(file)


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
    if o.context == 0:
        o = replace(o, context=m.max_context)
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
            f"{p.checkpoint_bytes / GIB:.1f}; working set {o.host_working / GIB:.1f}, an estimate"
            f"{f'; Engram cache and staging rings {p.host_caches_bytes / GIB:.1f}' if p.host_caches_bytes else ''}), "
            f"over the {limit / GIB:.1f} GiB {_limit_name(o)}. {_what_fits(m, o, limit)}"
        )
    return p


NGRAM_MARGIN_SHARE = 0.5
"""The share of the host's memory, at least 8 GiB, left free after the n-gram table moves into RAM: the unlocked arena
takes the host's free 2 MiB blocks, and with too few of them the host moves and swaps its small pages, which stops the
cards (docs/measurements/MEASUREMENTS.md#2026-11-04-ngram-room)."""


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
    or with an SSD tier takes only what is spare, so a smaller model file is the remaining way."""
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


def _fill(budget: int, sizes: Sequence[int], caps: Sequence[int]) -> tuple[int, ...]:
    """Whole experts per layer for `budget` bytes: each layer the same bytes, and what a layer cannot use, once it
    holds all `caps` of its experts, goes to the others. `sizes`: bytes of one expert in each layer."""
    out = [0] * len(sizes)
    left = budget
    order = sorted(range(len(sizes)), key=lambda i: caps[i] * sizes[i])
    for done, i in enumerate(order):
        out[i] = min(caps[i], left // (len(sizes) - done) // sizes[i])
        left -= out[i] * sizes[i]
    return tuple(out)


def _arena_budget(m: ModelMemory, o: PlanOptions, pinned_outside_arena: int) -> int:
    """What a model with an SSD tier may give its arena: the reserved huge pages' whole pieces, and what the host can
    spare; the host's part is 0 when it has none to spare, or is not known."""
    reserved = (o.hugetlb_pool or 0) // ARENA_PIECE * ARENA_PIECE
    limit = host_limit(o)
    if limit is None:
        return reserved
    ngram = m.ngram_bytes if o.ngram_in_ram else 0
    caches = m.engram_cache + m.ssd_staging_ring + m.prompt_host_ring
    return reserved + max(0, limit - pinned_outside_arena - o.host_working - ngram - caches - SSD_TIER_HEADROOM)


def _layer_experts(
    m: ModelMemory, o: PlanOptions, card_free: tuple[int, ...], pinned_outside_arena: int
) -> tuple[tuple[tuple[int, ...], ...], Tiers]:
    """What each card could hold a layer, from the same bytes a layer, and the tiers the smallest budget leaves."""
    sizes = m.layer_expert_bytes or (m.expert_bytes,) * m.layers
    in_layer = m.experts // m.layers
    shares = [-(-b // m.cards) for b in sizes]
    on_cards = tuple(_fill(free, shares, [in_layer] * m.layers) for free in card_free)
    # Every cached expert takes the same share of each card, so the tightest card's fill fits all of them. A per-layer
    # minimum across the fills can hold less, since a fill's carry moves between layers.
    vram = on_cards[card_free.index(min(card_free))]
    left = tuple(in_layer - v for v in vram)
    arena = _fill(_arena_budget(m, o, pinned_outside_arena), sizes, left) if m.ssd_tier else left
    ssd = tuple(n - a for n, a in zip(left, arena, strict=True))
    return on_cards, Tiers(tuple(sizes), vram, arena, ssd, m.routes_per_row, m.hit_curve)


def _plan(m: ModelMemory, o: PlanOptions) -> MemoryPlan:
    if o.slots < 1:
        raise PlanError("at least one slot")
    if not 0 < o.context <= m.max_context:
        raise PlanError(f"context {o.context} outside 1..{m.max_context}")
    vram = o.usable_vram
    if vram is None:
        raise PlanError("no card was read and --card-memory is not set")
    draft = m.draft_per_card if o.draft else 0
    pages_per_slot = -(-o.context // PAGE_CELLS)
    pages = -(-o.kv_tokens // PAGE_CELLS) if o.kv_tokens else o.slots * pages_per_slot + o.cached_prefix_pages
    if o.working_per_card is not None:
        working = o.working_per_card
    else:
        working = m.working_per_card + (m.working_of(o, pages * PAGE_CELLS) if m.working_of else 0)
    weights = m.weights
    per_sequence = reserve_per_sequence(m, o.context)
    reserve = o.slots * per_sequence
    card_free = tuple(vram - w - draft - working - reserve - m.prompt_staging_per_card for w in weights)
    free = min(card_free)
    if free < 0:
        staging = f", prompt staging {m.prompt_staging_per_card / GIB:.2f}" if m.prompt_staging_per_card else ""
        raise PlanError(
            f"the cards are {-free:,} B short before any expert: each has {vram / GIB:.2f} GiB, and the weights take "
            f"{max(weights) / GIB:.2f}, the draft layer {draft / GIB:.2f}, working memory "
            f"{working / GIB:.2f}{staging} and {o.slots} x {o.context:,} positions {reserve / GIB:.2f}."
        )
    if 0 < o.kv_tokens < o.context:
        raise PlanError(f"a KV pool of {o.kv_tokens:,} tokens holds no sequence of {o.context:,}")
    # The host holds every page's K and V (the core's kv_host), and each slot's draft ring.
    streamed = pages * PAGE_CELLS * m.kv_cell_host + o.slots * m.draft_kv_host
    checkpoints = o.checkpoints * m.checkpoint_bytes
    on_cards: tuple[tuple[int, ...], ...] = ()
    tiers: Tiers | None = None
    staging_per_card = 0
    if m.layer_expert_bytes or m.ssd_tier:
        on_cards, tiers = _layer_experts(m, o, card_free, streamed + checkpoints)
        cached = sum(tiers.vram)
        arena = tiers.bytes_of(tiers.arena)
    else:
        share = -(-m.expert_bytes // m.cards)
        stage = m.staging_per_uncached
        in_layer = m.experts // m.layers
        if free < in_layer * stage:
            raise PlanError(
                f"the cards are {in_layer * stage - free:,} B short of the prompt path's staging: "
                f"{in_layer * stage / GIB:.2f} GiB with no expert cached, and {free / GIB:.2f} GiB is free."
            )
        # n experts a layer take n x layers x share + (in_layer - n) x stage.
        cached = min(m.experts, (free - in_layer * stage) * m.layers // (share * m.layers - stage))
        staging_per_card = (m.experts - cached) * stage // m.layers
        arena = (m.experts - cached) * m.expert_bytes
    return MemoryPlan(
        options=o,
        cards=m.cards,
        usable_vram=vram,
        layers=m.layers,
        weights_per_card=max(weights),
        draft_per_card=draft,
        working_per_card=working,
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
        card_weights=weights,
        card_free=card_free,
        card_layer_experts=on_cards,
        tiers=tiers if m.ssd_tier else None,
        engram_cache_bytes=m.engram_cache,
        ssd_staging_bytes=m.ssd_staging_ring,
        prompt_host_ring_bytes=m.prompt_host_ring,
        prompt_staging_per_card=m.prompt_staging_per_card + staging_per_card,
    )
