# ADR-0065: a short chunk writes its experts through a mapping of the staging buffer

**Status:** draft, 2026-11-05; not accepted. Changes how [ADR-0046](0046-a-short-chunk-fetches-the-experts-its-router-picks.md)
item 1 makes its copies; which experts it copies, and when, stay.
**Revisit when:** the runtime's own copies from the arena reach the speed of a mapping, or a card's link is no longer
what limits a mapping's speed.

## Context

- A prompt chunk of 1,024 rows or fewer spends most of its time on these copies. On one card with 137 experts cached a
  layer, a 79-row chunk spent 700 of 806 ms in them, 212 rows 1,186 of 1,281 ms, and 550 rows 1,456 of 1,559 ms
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-staging-through-a-mapping)).
- The runtime makes a copy on an immediate list with the CPU, one at a time: 123 µs for a one-card expert of 2.05 MB,
  16.7 GB/s, and 14-19 GB/s in all with 1 to 8 threads on lists of their own.
- A device allocation made with a dma-buf export can be mapped with `mmap`. The CPU's `memcpy` into that mapping moves
  22.4 GB/s, 94 µs an expert, the same for 1 to 6 threads.

## Decision

1. `fetch_experts` writes an expert with `memcpy` through the CPU's mapping of the staging buffer, then at the end of
   the layer runs a full fence and reads the buffer once before it signals the layer's staging event.
2. The load writes the whole buffer once from the driver, so the buffer is in its place on the card, then touches each
   of its 4 KiB pages through the mapping, so no chunk takes the pages' first-write faults.
3. Where the system will not export or map the buffer, `fetch_experts` copies through the immediate list as before.

## Why

- First token on one card: long-gen 0.77 → 0.64 s, gpqa-1 1.26 → 1.04 s, decode level
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-staging-through-a-mapping)).
- A read of the card's memory returns only after the writes before it have arrived, so the card cannot see the event
  before the data. A fence alone sends the writes on their way and does not say they have arrived. The fence is a full
  one (`mfence`): a store fence orders only writes, so the read could go out ahead of the last writes.
- Pages touched before the card first uses the buffer move with it, and the first chunk then takes the faults again:
  long-gen's first token was 1.46 s that way.

## Options not chosen

- More threads on the immediate list: the runtime's copies do not scale with threads (above).
- More threads through the mapping: 22.4 GB/s is the same for 1 to 6 threads, so more threads add nothing.
- The CPU computing the uncached experts few rows pick, with no copy: see the record for what it adds on top of the
  mapping.

## Consequences

- A start takes about 0.5 s longer on one card, for the driver's write and the touch of the 1.91 GiB buffer.
- The runtime keeps the exported descriptor and closes it when the memory is freed; the memory returns on free
  after the mapping is removed, so sleep and wake hold nothing back.
