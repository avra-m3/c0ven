# ADR-0010: several requests - pipeline two across the cards first, then batch

**Status:** accepted, 2026-10-09. Item 1 superseded by [ADR-0023](0023-two-cards-tensor-split.md) on 2026-10-12:
one request uses both cards, so there is no second request to pipeline onto the other card. Items 5 and 6 are made
precise by [ADR-0017](0017-scheduling-several-requests.md).
**Revisit when:**

- the pipelined cards balance badly;
- batching gains little because the union of experts grows almost linearly with the batch.

## Context

With a layer split, one request uses one card at a time, and each card idles about half of every window. In
Strata's measurements, a 5080 + 3090 split decoded no faster than the 5080 alone: 83.8 against 83.1 tokens/s.

## Decision

1. **First, two requests in flight, one per card:** card 0 runs request A's window while card 1 runs B's, then they
   swap. This needs:
   - per-sequence state;
   - a host thread per card;
   - doubled hand-off buffers;
   - one CPU-expert channel per card, each pinned to one CCD.

   It needs no batched kernels.
2. **Then batched windows:** the verify windows of several sequences go in one pass, inside each pipelined group.
   The draft length per sequence shrinks as the batch grows, by measurement.
3. **Every per-token kernel takes a sequence index from its first version,** so batching does not mean rewriting
   kernels.
4. **Sequence slots are carved at start,** sized from the KV reserve ([ADR-0008](0008-kv-cache.md)). There are no
   allocations on the token path.
5. **Long prompts run in chunks between decode passes,** so a long prompt slows the other requests without stopping
   them.
6. **The server:**
   - a scheduler with N slots;
   - request ids on every line of the engine's protocol;
   - requests wait in a queue when slots are full;
   - cancelling one request frees its slot alone.

## Why this order

- **Pipelining first:** it needs no kernel changes, and should give the larger share of the gain (*estimate*: close
  to 2x the total output of one request).
- **Batching second:** it changes what is slow (more experts per layer, more CPU work), so it can only be judged
  against a tuned single-sequence engine.

## Consequences

- **The concurrency floor** is llama.cpp with several requests ([ADR-0013](0013-performance-floor.md)).
