# ADR-0070: the n-gram table takes the reserved huge pages the arena leaves

**Status:** draft. 2026-11-08. Supersedes item 3 of [ADR-0066](0066-the-arena-may-take-reserved-huge-pages.md).
**Revisit when:** a run with the n-gram cache in reserved huge pages is timed against one without it, or the hit
rate of a cache that size is measured.

## Context

- [ADR-0066](0066-the-arena-may-take-reserved-huge-pages.md) chose reserved huge pages for the expert arena only to
  stop the stall when the kernel moves a small arena page under a copy binding. It turned down the pinned buffers
  because they have no such stall. It did not weigh reserved pages that sit idle.
- Reserved huge pages (`vm.nr_hugepages`) are outside `MemAvailable` and serve no other use on the host while they
  are reserved. The operator sizes the pool once, for the largest arena of the deployments the host runs, so a
  deployment with a smaller arena leaves pages free. One such deployment, on two cards with 3 slots of 262,144
  positions, holds 17.14 GiB of arena in a pool of 36 pages of 1 GiB, and leaves 17 GiB free
  ([record](../measurements/MEASUREMENTS.md#2026-11-08-ngram-cache-in-file)).
- The arena grows after the load as the KV pool takes card memory from the expert cache, down to the experts a layer
  the load keeps room for with the whole pool backed (`experts_floor`): 16,642 MiB at load and 22,080 MiB after two
  255,028-token prompts ([record](../measurements/MEASUREMENTS.md#2026-10-25-arena-resident-and-tail-drop)).
- The n-gram table (28.8 GB for the IQ3_S file, 320,001,536 rows of 90 B) stays in the file unless the host has
  room for it with half its memory left free ([ADR-0057](0057-the-engine-reads-the-host-it-runs-on.md) item 6). That
  margin protects the free 2 MiB blocks the unlocked arena takes; without it the host moved and swapped small
  pages and stopped the cards ([record](../measurements/MEASUREMENTS.md#2026-11-04-ngram-room)).
- Read from the file, the table's rows go through a direct-mapped cache of 1,048,576 rows (about 100 MB, 0.33% of
  the rows). In the deployment above it serves 77% of the rows asked for, which leaves about 2,860 reads from the SSD a
  second at about 76 µs each ([record](../measurements/MEASUREMENTS.md#2026-11-08-ngram-cache-in-file)). The device waits for the rows before layer 1
  ([ADR-0015](0015-ngram-table-on-the-ssd.md) item 3).

## Decision

1. With `--hugepages` and `--ngram-ram auto`, the n-gram table takes the free reserved huge pages that the arena
   leaves at its largest. The arena's claim comes first: its bound is the plan's experts a layer, less the fit
   report's tolerance (5%, at least 8 experts), and its pieces at that bound and one more.
2. The table goes in RAM whole when those pages, and the host's memory under the margin, hold it. Its first part is
   in the reserved pages and the rest in the process's own memory, at one range of addresses.
3. Otherwise the row cache fills those pages, as many rows as they hold, when that is more than the default cache.
   Its pages are faulted in at the load. The table stays in the file.
4. The margin of ADR-0057 item 6 applies only to the part of the table outside reserved huge pages.
5. The memory plan and the fit report count the n-gram's reserved pages apart from `MemAvailable` and `memory.max`,
   and say where the table is and how many rows its cache holds. The C ABI's config gains `ngram_huge_bytes` and its
   info `ngram_cache_rows` and `ngram_huge_bytes` (ABI version 20).
6. `--ngram-ram on` and `off`, and every run without `--hugepages`, are as before.
7. The pinned buffers stay with the driver.
8. A sleep keeps the table and its cache, as it keeps the arena; `--sleep-mode exit` gives them back
   ([ADR-0032](0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md)).

## Why

A cache beats a fixed part of the table. The table's row ids are hashes of token n-grams, so the rows a prompt
reads are spread evenly over the table, and a fixed part holding a share of the rows serves about that share of the
reads: 44% for the 13 GiB the deployment above leaves at its arena's largest (*estimate*: the plan for those options
on the mock card, `tests/host/test_memory_plan.py`). The default cache already serves 77% with 0.33% of the rows,
so the reads are skewed towards rows read before, and a cache of 142 million rows, 136 times larger, keeps more of
them. Its hit rate is not measured. A fixed part also
needs a profile to choose its rows, and its pages filled at load.

The margin protects the host's free 2 MiB blocks and its small pages. A table in reserved pages takes neither.

The cache's tags are 64-bit and its slots are counted in 64 bits, so its size has no limit below the table's. One
thread reads it, as before: a row costs one lookup at a slot, a tag compare and a 90 B copy, whatever the cache's
size. In 1 GiB pages the cache needs few TLB entries.

The pinned buffers (the KV host copies and checkpoints, the swap buffer, the image encoder's 16-bit weights and the
step buffers) come from `zeMemAllocHost`. The driver allocates and pins them, so they never move, and the cards read
them without a binding. Moving them into reserved pages would need the driver to import memory it did not allocate,
the binding path ADR-0066 kept for the arena only, and nothing they do waits on a page fault. The n-gram readers'
buffers are 8 KiB a thread.

## Options not chosen

- **A fixed part of the table in reserved pages.** It serves about its share of the reads, below the cache's rate.
- **The cache in reserved pages under `--ngram-ram off`.** `off` keeps its meaning, the table in the file with the
  default cache, for comparing against.
- **The core sizing the arena's claim after its load, from its own `experts_floor`.** Python owns the plan
  ([ADR-0022](0022-two-processes-python-and-cpp.md)), and an arena past its claim takes ordinary memory as before.

## Consequences

- An operator sizes `vm.nr_hugepages` for the arena at its largest and as much of the n-gram table as the pages
  should hold: 27 GiB more holds the IQ3_S file's table whole.
- The load spends a few seconds clearing the cache's pages.
- No speed is claimed until a timed run measures it against main.
