# ADR-0048: the KV split's blocks move beside the steps

**Status:** draft, 2026-10-24; not accepted. Supersedes the last clause of
[ADR-0045](0045-the-kv-pool-and-the-expert-cache-share-each-cards-memory.md) item 5 ("moves the blocks between plans").
**Revisit when:** a list reads cache slots its maps do not name, or a driver makes `zeVirtualMemUnmap` or
`zeVirtualMemMap` wait for or preempt a running list.

## Context

- When the KV pool takes blocks from the expert cache ([ADR-0045](0045-the-kv-pool-and-the-expert-cache-share-each-cards-memory.md)),
  the engine thread copied the experts that stay down out of the blocks and waited for each layer's copies, wrote the
  maps, then unmapped each block from the cache and mapped it into the pool. With 4 slots, a 262,144-token context and
  a 524,288-token pool that blocked it 14.3-31.7 ms a change, 5 times a run, and giving the pages of four ended
  requests back 34.8 ms ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-20-kv-pool-and-expert-cache-share)).
- A decode list may read any slot the maps name, so the maps change only between plans. Once they no longer name a
  block's slots, no list reads that block; no list reads the pool's pages above those in use either.
- Moving 2 MB blocks on one thread while the same card runs kernels or copies neither stalls nor faults them: work on
  the moving range ran 1-2.5% slower while the moves lasted, other work at its full rate, and every read was right
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-24-block-moves-beside-running-work)).

## Decision

1. A change of the split appends every layer's copies of the experts that stay and the maps to the cards' copy lists,
   and waits for them once (`apply_evict`).
2. The blocks' unmaps and maps are queued as the core's bookkeeping moves the blocks (`map_cache`, `grow_kv`,
   `shrink_kv`). After a change and after a give-back, a thread makes them beside the next steps (`run_vm`); the
   engine thread goes on.
3. Blocks that a copy or a plan needs at once are mapped at once: those a refill copies experts into, those of a plan
   that waits for its pages (`kv_need`), and the load's.
4. One batch of maps runs at a time, in order. The bookkeeping and the choice of later changes see the blocks where
   the batch puts them; a plan that needs pages the batch has not yet mapped waits for it.

## Why

The thread's moves cost the steps about 1% for the few milliseconds they last (*estimate*: a change moves 29-72 blocks
a card at 55-84 µs each, beside steps that run 1-2.5% slower then,
[MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-24-block-moves-beside-running-work)), against 14.3-31.7 ms of the engine
thread before. The copies left on the engine thread are at most 144 of about 1 MB a card (*estimate*: under 1 ms at
197 GB/s). The gain is in latency: a first token or a writer's step that falls on a change no longer waits for it.

Measured over 7 loads an arm at full context and full load: a change blocks the engine 1.0-3.6 ms instead of
14.5-37.8 ms, and no prompt or decode rate changed beyond the spread of the loads
([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-25-kv-moves-beside-the-steps)).

## Options not chosen

- Vacating the blocks a round ahead, the experts that stay copied with that round's swaps: it saves only the copies'
  wait, under 1 ms, and leaves the slots empty for a round.
- Keeping the moves between plans and only waiting once for the copies: the moves themselves are most of a change's
  time.

## Consequences

- The engine's log gives a change's time on the engine thread, and a second line the time of its blocks' moves beside
  the steps.
- `kv_own_` counts the pages backed once the queued maps are made, `kv_mapped_` those backed now.
- `--deterministic` turns sharing off and is unchanged.
