# ADR-0023: two cards - every layer split across both, the CPU computes the misses

**Status:** accepted, 2026-10-12; item 5 added 2026-10-13. Supersedes
[ADR-0009](0009-two-cards.md), and item 1 of [ADR-0010](0010-several-requests.md). Item 2's "No peer-to-peer"
superseded by [ADR-0030](0030-exchanges-into-the-other-card.md). Item 6 superseded in part by
[ADR-0035](0035-the-arena-in-the-engines-own-memory.md): the arena is the engine's own memory.
**Revisit when:** the engine's timing ([DESIGN.md, section 7](../DESIGN.md#7-timing-and-debugging)) shows the CPU's misses, or the
exchanges, costing more than streaming or a layer split would.

## Context

- **ADR-0009's revisit condition is met.** A card-to-card exchange through host memory, with a kernel on each card
  waiting on the other's flag, takes 10.0-10.5 µs at 1-4 tokens, far under its 0.1 ms
  ([record](../measurements/MEASUREMENTS.md#2026-10-12-card-to-card)).
- **Weights are most of a token's work:** about 5.1 GB read per token (*estimate*, from the file's tensor sizes).
  Split across both cards, each reads half.
- **A decode step shaped as a tensor split** was measured with stand-in kernels timed to the
  [kernel spike](../measurements/MEASUREMENTS.md#2026-10-11-kernel-spike)'s rates, the card-to-card exchanges, and the
  CPU computing the misses with ggml ([record](../measurements/MEASUREMENTS.md#2026-10-13-design-inputs)). One
  sequence:

  | ms per token | Hit rate 0.931 | 0.973 |
  | --- | ---: | ---: |
  | One card, CPU computes the misses | 15.56-15.60 | 14.92-14.97 |
  | Both cards, CPU computes the misses | 10.99-11.28 | 9.76-9.86 |
  | Both cards, the cards stream the misses | 11.19-11.28 | 10.39-10.43 |

- **Under a competing RAM load** the CPU path slowed 1.6-2.2 times and streaming 10-27%, in the same record.

## Decision

1. **Every layer is split across both cards.** Each card holds and computes half of every layer: weights, expert
   cache, KV and state heads, buffers. Nothing a card owns is copied to the other. Each cached expert is split by
   its rows, half on each card, so both cards hold the same experts and work on every expert of a token at once. A
   missed expert is computed whole by the CPU. The indexer's keys are split by position:
   each card scores half the blocks and the cards exchange their best 512 candidates, one more exchange per
   attention layer ([DESIGN.md, section 4](../DESIGN.md#the-indexer-on-two-cards-split-by-position)).
2. **The halves are combined by exchanges through host memory,** twice per layer, as the card-to-card record measures them. No peer-to-peer.
3. **The CPU computes the expert misses** in place in the host arena, with ggml's kernels
   ([record](../measurements/MEASUREMENTS.md#2026-10-11-cpu-bar)). The cards do not stream misses.
4. **No card is kept busy to hold its clock up.**
5. **The host arena holds only the experts the cards do not cache.** Every expert is in exactly
   one place: on the cards (half on each) or in the arena (whole). A swap moves an expert both ways on the copy
   engines: the one leaving the cards goes back to the arena, its two halves from the two cards, and the one
   entering comes from the arena and frees its place. An arena of every expert (46.84 GiB) does not fit in the host
   memory that can be pinned ([MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md#ram-for-the-tensor-split)).
6. **Kept from ADR-0009:** one Level Zero context for both cards; device memory for what a card owns, host memory
   for what both cards or the CPU touch, never shared memory; per card an in-order compute queue, an in-order copy
   queue and its own host thread.

## Why

- Both cards work on every token, which almost halves the step: 8.25 ms with no misses against 13.66 on one card,
  in the same record.
- The CPU path is the simplest, and on a quiet host it matched or beat streaming.

## Options not chosen

- **The layer split (ADR-0009).** One request uses one card at a time.
- **The cards streaming the misses.** Held up better under RAM load, but slower on a quiet host at 0.973; kept as a
  later optimisation (splitting misses between both).
- **Keeping a card busy to hold its clock up.** A waste of power for a cost not measured.

## Consequences

- **[ADR-0010](0010-several-requests.md)'s item 1** (two requests pipelined, one per card) no longer applies: one
  request already uses both cards. Batching (its item 2 onwards) stays.
- **How the cards share the work** is in [DESIGN.md, section 5](../DESIGN.md#5-two-cards-every-layer-split-across-both).
- **RAM bandwidth stays the shared limit:** the CPU workers read the arena while the cards exchange through host
  memory.
- **Two exchanges per layer, plus one per attention layer** for the indexer's candidates: 108 per token.
- **Pinned host memory is 10.8-26.4 GiB** (one 32K sequence to four at 262K, arena plus streamed KV), not 47-60 GiB.
  Four sequences at 262K reach the edge of what can be pinned.
- **A swap costs two copies, not one,** and the arena's free places must be tracked as experts move.
- **[ADR-0018](0018-one-file-iq3s.md)'s RAM condition** (the arena and the streamed KV do not fit in free RAM) was
  met, and is answered by item 5 rather than by changing the file.
