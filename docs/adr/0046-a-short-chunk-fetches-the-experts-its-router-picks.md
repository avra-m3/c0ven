# ADR-0046: a short prompt chunk fetches the experts its router picks

**Status:** draft, 2026-10-21; not accepted.
**Revisit when:** compute-runtime stops making a copy from host memory it did not allocate with the CPU on an
immediate list ([LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md) row 31), the cards cache so many experts that a chunk's staging
hides behind its products, a chunk of more than 1,024 rows should fetch as well, or a chunk of 500-1,024 rows is
measured slower than one that stages every uncached expert.

## Context

- Each card stages, for every prompt chunk, every expert it does not cache, and the chunk's products wait for them.
  With 330 experts a layer cached, the 182 uncached ones take each card about 0.43 s a chunk whatever its length (48.8
  µs an expert), and a 247-351-token prompt reaches its first token in 0.484 s
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-20-kv-pool-and-expert-cache-share)).
- A short chunk routes to few of them. Chunks of 242-346 rows route to 44-70 uncached experts a layer (the mean over
  the 48 layers), 498-577 rows to 58-82 and 971 rows to 84
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-21-short-chunk-fetch)).
- Which experts a chunk routes to is known only after each layer's router has run, a few milliseconds before the
  products that read them.
- A card's copy engine has one queue (row 10). A second queue at index 0 is a second handle on one ring: its list
  waits behind the first queue's list while that waits on an event; index 1 is refused. So copies chosen after the
  router cannot pass a recorded copy list that waits for a later layer.
- From the anonymous arena ([ADR-0035](0035-the-arena-in-the-engines-own-memory.md)) a recorded copy costs a binding
  (row 31), so recording a list per expert costs 114-144 µs a list, 1.0-1.3 s a card for every uncached expert at
  load. On an immediate list the runtime makes the same copy with the CPU: the call returns when it is done, 48-57 µs
  an expert (1.06 MB) with one card copying and 50-79 µs with both, against 49 µs with both on the copy engine, and it
  runs while the copy engine's queue waits on an event
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-21-short-chunk-fetch)).

## Decision

1. A prompt chunk of 1,024 rows or fewer stages its experts from the host. A thread a card copies each layer's
   uncached experts into their staging positions through an immediate list on the copy engine's ordinal: the hottest
   first by the swap counts, from when the chunk frees the layer's slot until the layer's router has run, then every
   other uncached expert the router picked. Then it signals the layer's staging event.
2. The chunk's list copies each layer's routing counts to host memory after the router, with an event, and runs the
   cached experts' and the shared expert's products before it waits for the staging event.
3. A longer chunk keeps the copy list recorded at load, which stages every uncached expert.
4. The staging positions and maps stay as the recorded list has them. Every expert the chunk routes to is staged
   before its products, so the products, and the tokens, do not depend on which experts went ahead of the router.

## Why

- A chunk waits only for the experts it reads. Chunks of 108-127 rows took 199-218 ms and 242-346 rows 272-365 ms,
  where staging every uncached expert takes about 0.43 s of copies, and a 247-351-token prompt's first token came
  0.486 → 0.376 s ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-21-short-chunk-fetch)).
- The experts copied before the router has run are the ones the swap counts make most likely, so fewer are left for
  after it: at 242-346 rows, 12-31 of the 34-46 a layer copied ahead were routed to.
- The CPU's copies need no list recorded per expert, no binding, and nothing recorded again when a swap or the cache's
  growth moves an expert's staging position.
- 1,024 rows: a 971-row chunk took 490-495 ms. One prompt was slower: edit-py (584 tokens) reached its first token in
  0.519 s against 0.503 (medians; the ranges overlap).

## Options not chosen

- A recorded list per uncached expert, run as the router's subset: about 8,700 lists a card (181-182 uncached experts
  in each of 48 layers), 1.0-1.3 s a card to record at 114-144 µs a list and 7.5-17 KB of host memory each, recorded
  again each time a swap or the cache's growth moves an expert's staging position.
- Copies chosen after the router on a second copy queue: it shares the first's ring (above).
- A fixed number of hot experts staged ahead by a recorded list: the swaps take the hottest uncached experts into the
  cache, so the experts a recorded list stages ahead would turn cold, and the time before the router changes with
  the chunk's rows.

## Consequences

- Two host threads run during each short chunk, and the CPU's copies use host memory bandwidth the copy engines would
  have used.
- A 64-token prompt piece still runs as decode windows; against a chunk that fetches, that choice is not measured.
