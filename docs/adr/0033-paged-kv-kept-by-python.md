# ADR-0033: the KV cache in pages, the page table kept by Python, checkpoints of the fixed state only

**Status:** draft, 2026-10-17; not accepted.
Would settle items 2 and 6 of [ADR-0027](0027-the-c-abi-as-the-core-was-optimised.md), item 6 in part.
**Revisit when:** a measurement shows the page table costing decode or prompt time, or KV streaming needs pages
in host memory that this table cannot name.

## Context

- **The design pages the KV.** [DESIGN.md, section 4](../DESIGN.md#4-sequence-state-the-kv-cache-and-the-rest):
  Python keeps the page table, shares prefix pages and sends page changes in each plan; the layout is one header the
  core and the kernels share; the core reads and writes the pages. A checkpoint holds only the fixed state, 118,038,528
  B: the 36 DeltaNet layers' state and the PLE history.
- **Versions 3 and 4 of the ABI do not.** Each slot keeps its whole context's KV, and a checkpoint copies the KV up
  to its position, so `engine_open` pins every checkpoint at the whole context's size: about 4 GiB each at 262,144
  tokens. Eight of them do not fit beside the arena
  ([record](../measurements/MEASUREMENTS.md#2026-10-16-engine-start-and-quit)), so those versions default to 2.
- **Python's half exists.** `engine/seqstate/` keeps pages, reference counts, reservations and checkpoints, and the
  scheduler admits by page count; with versions 3 and 4 the native adapter drops the page changes
  (ADR-0027, item 2).
- **KV streaming needs a table.** ADR-0008 item 3 keeps only the most-read cells in VRAM. Which cell lives where is a
  per-page fact the kernels must look up.

## Decision

1. The KV cache is paged. A page is 4 cells, one indexer block. Each attention layer of a card has one pool of pages
   holding K and V for the card's heads; a page id names the same page on both cards. The layout is in
   `core/models/qwen4exp/layout.h`.
2. Python keeps the page table. A step plan carries page changes, applied in order before its prompt pieces: map a
   slot's logical page to a page, unmap a logical page and every page after it, or copy the first 1-3 cells of a page
   and map the copy. The core keeps a copy of each slot's row on each card and writes it only for plans that change
   it.
3. On two cards logical page i maps only to a page p with p % 2 == i % 2. The indexer's keys stay split by position
   ([DESIGN.md, section 4](../DESIGN.md#the-indexer-on-two-cards-split-by-position)) at no extra memory: page p's raw
   and pooled keys live on card p % 2.
4. A slot reads and writes only through its mapped pages. A slot that reaches a position past them waits and says so
   once (`ENGINE_EVENT_STALLED`), and a draft window stops short of them.
5. A checkpoint holds the fixed state only. Python keeps the pages of the checkpoint's positions and maps them into the
   slot it restores; the page the checkpoint ends inside is copied. `checkpoint` and `restore` still wait for the
   queued plans: that part of ADR-0027 item 6 stays open.
6. The pool's default is the slots' whole contexts, the VRAM versions 3 and 4 used. The memory plan adds no pages for
   prefixes by default (`PlanOptions.cached_prefix_pages`), and counts the checkpoints in its pinned total at their
   new size.
7. The scheduler maps a decoding slot's pages at least 64 cells ahead, 256 cells at a time, so few plans carry
   changes.
8. The draft layer's KV is not paged. Its kernels are given no table and read each slot's own run of pages.

## Why

The table costs no time that a decode or a prompt shows
([record](../measurements/MEASUREMENTS.md#2026-10-17-paged-kv)). On doc-32k at 64K, one sequence, ABBA with 6 runs an arm: decode
18.62 ms per step against 18.48 ms before, the ranges overlapping; with drafting 76.10 against 75.24 tokens/s; the
prompt 2,243.5 tokens/s in both. Every run gave the same tokens. Traced, the five kernels that read the table add
11-14 µs to an 18.3 ms step: one read of the table per cell or block they touch, from a few kilobytes per slot.

Checkpoints of the fixed state only take 118,038,528 B each at any context, against about 4 GiB at 262,144 tokens
before; that is what lets the engine's default count of checkpoints fit beside the arena.

## Options not chosen

- **The table in host memory, read by the kernels.** The indexer's scoring reads one entry per block per row: at
  262,144 tokens, 65,536 reads over PCIe per attention layer.
- **Page changes by a call of their own.** Plans are numbered and the adapter already knows when a plan naming a slot
  has run (ADR-0027, item 13); a separate call would need its own ordering and its own completion.
- **The indexer's keys split by page id instead of position.** The scoring kernels would need the reverse map, from a
  card's pages back to the slot's blocks.
- **Every page's indexer keys on both cards.** That frees the parity rule but doubles the keys: 214 MB per card for one
  sequence at 262,144 tokens, about 209 experts.
- **A kernel in each recorded step that applies pending changes.** It costs a launch on every step, with changes or
  not.

## Consequences

- `engine_api.h` is version 5: `engine_page_change`, the plan's `npages` and `pages`, `kv_pages` in the config and the
  info, `pages_per_slot`, `ENGINE_EVENT_STALLED`. `engine/api/abi.py` mirrors it, and the native adapter passes the
  scheduler's page changes on instead of dropping them.
- A checkpoint is 118,038,528 B of pinned memory at any context: 8 of them are 0.94 GB. The engine process's and the
  plan's default goes back to 8.
- A wake starts every slot with no pages mapped, as a sleep drops sequence state
  ([ADR-0032](0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md), item 9).
- `Model` keeps its own default table (each slot its own run of pages), so the core's tests that drive the model
  directly run as before.
- KV streaming can mark pages as host-resident in the same table.
