# ADR-0079: a swap round reads its outgoing experts past the page cache, two reads at a time

**Status:** draft. 2026-11-12. Supersedes item 7 of
[ADR-0032](0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md) in part (the swaps read experts through the
mapping).
**Revisit when:** the model file and the n-gram table sit on different disks, or a round's reads at two at a time take
longer than the plans between rounds.

## Context

- A swap round gives each expert that leaves the cards a place in the arena. Its only copy in host memory is the model
  file, so the swap thread reads it from there into a spare piece, beside building the incoming experts' halves
  (`core/models/qwen4exp/swap.cpp`). The next round waits for that thread.
- The load reads the weights with `O_DIRECT` and gives the file's pages back (ADR-0032, item 7). The swap thread read
  through the mapping instead, and gave each expert's pages back once read, since a page cache that grows makes the
  host reclaim and move the engine's buffers
  ([the record](../measurements/MEASUREMENTS.md#2026-10-17-decode-stalls-eviction)).
- Through the mapping, each page fault waits for a read of its own: about 640 major faults for 64 experts, and the
  thread took about 100 ms where one writer's rounds came a median 100 to 109 ms apart (`SwapCadence`). One request
  on two cards waited for it 59 to 109 ms over 300 tokens
  ([the record](../measurements/MEASUREMENTS.md#2026-11-12-swap-reads-past-the-page-cache)).
- The n-gram table's rows come from the same disk, in 4 KiB `O_DIRECT` reads that each step waits for, about 62 for
  each draft round of one request ([ADR-0015](0015-ngram-table-on-the-ssd.md)).

## Decision

1. The swap thread reads each matrix of an outgoing expert, and of a planned eviction's victims, with one `O_DIRECT`
   read into a buffer its thread keeps, and copies it into the arena piece in the arena's layout, as before. Nothing
   reads an expert through the mapping, and nothing gives an expert's pages back. The mapping stays for the token
   embedding.
2. Two of the swap thread's tasks read, one matrix at a time each (`kFileReaders`); the other threads build the
   incoming halves meanwhile.
3. An eviction that the engine thread waits for reads on the engine's own pool, one read a thread.
4. The engine opens the model's files with `O_DIRECT` for its whole life. A filesystem that refuses it stops the load,
   as it does for the load's reads.

## Why

From traces of one and two requests on two cards, the swap thread's round of 64 experts and each round's wait for it,
over 300 tokens of one request
([the record](../measurements/MEASUREMENTS.md#2026-11-12-swap-reads-past-the-page-cache)):

| Reads | A 64-expert fill, median | Wait for the swap thread | A draft round |
| --- | --- | --- | --- |
| Through the mapping | 101 to 105 ms | 59 to 109 ms | 23.5 to 24.3 ms |
| `O_DIRECT`, one at a time | 95 to 97 ms | 1 and 27 ms | 23.8 to 24.7 ms |
| `O_DIRECT`, two at a time | 59 ms | none | 23.1 to 24.4 ms |
| `O_DIRECT`, four at a time | 44 to 52 ms | none | 24.8 to 25.2 ms |
| `O_DIRECT`, four at a time in 128 KiB reads | 84 to 99 ms | 0 and 22 ms | 23.8 to 24.0 ms |

Four reads at once end soonest, but each draft round is about 1.1 ms slower than through the mapping with the same
tokens, more over a request than the wait they remove. With the same four in 128 KiB reads the rounds are level again,
so the cause is most likely the n-gram table's small reads waiting behind large ones. Two at once end well inside the
time between rounds, and the rounds are as fast as through the mapping.

On two cards, in three ABBA rounds against reads through the mapping, one request wrote 2.5% ± 1.4 faster, two 0.4%
± 1.6, and one with an 8,050-token prompt 3.6% ± 0.7; first token and prefill were level. On one card the rounds come
180 to 230 ms apart, so the mapping's reads seldom held a round up (0 to 9 ms a request) and there is nothing to gain.

## Options not chosen

- Asking the kernel for all of a round's reads as soon as the swaps are chosen (`posix_fadvise(POSIX_FADV_WILLNEED)`)
  and reading through the mapping as before. The kernel reads only a readahead window of each range: the faults fell
  from about 640 to 570 a round and the waits to 53 and 54 ms.
- A round that is due while the swap thread works waits for a later plan. Nothing waits, but rounds come later and
  the cache follows less closely: two requests wrote 103 and 106 tokens/s against 108 to 112.
- Keeping a window of the file's pages in the page cache. No expert left the cards twice in a load (none of 1,455
  reads), so a window would hold pages nothing reads again.
- Reading an outgoing expert back from the cards. No disk read, but it needs the inverse of the cards' layout from
  both cards' halves, and copies on the cards' queues beside the steps.

## Consequences

- Each thread that reads keeps one widened matrix while it lives, about 0.9 MB (*estimate*: the largest expert's
  2,662,400 B in three matrices, ADR-0032): the swap pool's three threads and the engine pool's threads for the
  engine's life, the swap thread for its round.
- `GgufModel::release` gives the dense tensors' pages back after the load; no expert's pages enter the page cache.
- On one card, with the 8,050-token prompt, decode was 9.7% ± 4.9 slower over five ABBA rounds and its rounds 0.9
  ms slower at the median, with one reading task or two; the cause is not found
  ([the record](../measurements/MEASUREMENTS.md#2026-11-12-swap-reads-past-the-page-cache)).
