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
2. The load writes the whole buffer once through an immediate list, then touches each of its 4 KiB pages through the
   mapping, so no chunk takes the pages' first-write faults.
3. Where the system will not export or map the buffer, the load fails. There is no second way to stage the experts:
   on such a system, the earlier version is the one to run.
4. `fetch_experts` counts the page faults its thread takes while it writes a chunk's experts, and logs them when there
   are any.

## Why

- First token on one card: long-gen 0.76 → 0.64 s, gpqa-1 1.26 → 1.04 s, decode level
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-staging-through-a-mapping)).
- The full fence sends the CPU's writes out of its write-combining buffers (Intel SDM vol. 3A, "Buffering of Write
  Combining Memory Locations"). That says they are on their way, not that they have arrived. A read of the card's
  memory after the fence returns only once they have arrived, since PCIe does not let a read pass earlier writes on
  the same path (PCI Express Base Specification, "Transaction Ordering Rules"). So the card cannot see the event
  before the data. A store fence would not do: it does not hold back a later read, so the read could go out ahead of
  the last writes.
- This needs a write-combining or uncached mapping. xe maps a card's memory write-combining (`xe_ttm_io_mem_reserve`
  sets `ttm_write_combined`). In a write-back mapping the writes would sit in the CPU's cache, and neither the fence
  nor the read would send them.
- The immediate list's write into the buffer is a copy the runtime makes with the CPU
  ([LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md) row 31). Pages touched through the mapping before it fault again
  after it, and the first chunk then takes the faults: long-gen's first token was 1.46 s that way, against 0.76 s
  before this decision. Why that copy undoes the mapping's pages is not known; neither the Level Zero specification
  nor the records explain it. The order rests on that measurement alone, so a driver that changes it would bring the
  faults back. The fault count in the log says so when it happens: a build without the touch logged 140,476-146,226
  faults on its first chunk, and the build with it logs none.

## Options not chosen

- More threads on the immediate list: the runtime's copies do not scale with threads (above).
- More threads through the mapping: 22.4 GB/s is the same for 1 to 6 threads, so more threads add nothing.
- The CPU computing the uncached experts few rows pick, with no copy: [PR 133](https://github.com/avra-m3/c0ven/pull/133)
  measures what it adds on top of the mapping, and decides it there.

## Consequences

- A start and a wake each take longer by the touch. On one card at one slot of 262,144 the buffer is 1.91 GiB and its
  touch 224 ms; the start was 0.5 s longer in all (25.1 → 25.6 s), and the rest is not traced. On two cards at the
  same settings the buffer is 0.37 GiB a card and its touch 43 ms a card, one card after the other, for a gain that is
  level there. The path is not limited to one card: what decides the gain is how many experts the cards leave out,
  not the card count, and the cost scales with the same buffer.
- Another process that takes the card's memory pushes the buffer to host memory. The card's next use brings it back,
  and the CPU's next two writes fault on every page: about 0.75 s once, on the next short prompt; the bytes stay the
  same ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-staging-through-a-mapping)).
- The runtime owns the exported descriptor: it returns the same number each time it is asked and closes it in
  `zeMemFree`, so the engine never closes it
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-staging-through-a-mapping)). The memory returns on
  free after the mapping is removed, so sleep and wake hold nothing back.
