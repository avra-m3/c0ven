# ADR-0015: the n-gram table stays on the SSD, read row by row

**Status:** accepted, 2026-10-09. Item 6 superseded by [ADR-0018](0018-one-file-iq3s.md) on 2026-10-10: the engine
runs IQ3_S only, so the table in RAM is no longer a standing A/B arm. Superseded in part by
[ADR-0057](0057-the-engine-reads-the-host-it-runs-on.md) (draft): the table is read into RAM when the host has room
for it.
**Revisit when:**

- the SSD's 4 KiB random reads are too slow, or too uneven, to hide behind layer 0;
- or the RAM grows enough to hold the 28.8 GB table beside the largest format the engine runs.

## Context

- **The table is large.** It holds 320M rows of 90 bytes (IQ4_NL), 28.8 GB, and each token reads 16 rows on 16
  random pages.
- **RAM is short.** The table fits in RAM only beside Q2_0 (~84 of 92 GB), not beside IQ3_XXS or IQ3_S
  ([MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md)).
- **The page cache would hide the cost and the memory.** Reads through it would fill RAM with pages the engine does
  not count, and evict what it does.
- **Strata reads the table this way already.** It uses unbuffered 4 KiB reads (`O_DIRECT`) from a pool of 16
  threads, with a cache of rows already read (1M rows, ~95 MB). That served up to ~82% of reads on its test corpus
  (`include/strata/ngram/ple_reader.hpp`, `src/platform/direct_file.cpp`).
- **The SSD sits on its own CPU root port,** so the reads share no link with either card.

## Decision

1. **The table stays on the SSD.** It is read with `O_DIRECT` 4 KiB reads, and never enters RAM or the OS file
   cache.
2. **A cache of rows already read** sits in RAM. Its size is a start option, and the engine reports its hit rate.
3. **The reads never wait on the host in a window.** The engine starts a window's reads when its tokens are known,
   and the device waits for the rows just before layer 1 (the only layer that reads them), behind a flag in host
   memory ([ADR-0011](0011-cpu-gpu-signalling.md)).
4. **Decode reads go ahead of prompt reads.** A prompt chunk's reads may hold only a capped share of the reads in
   flight. The cap is set by measurement.
5. **The model file must be on a filesystem that honours `O_DIRECT`.** The engine refuses to start otherwise.
6. **The table held in RAM is a switch for measurement only:** the zero-cost reference the SSD path is measured
   against, on Q2_0.

## Options not chosen

| Option | Why not |
| --- | --- |
| The table in RAM, mapped and locked | 28.8 GB: fits only beside Q2_0. Kept as the A/B arm |
| Reads through the page cache | the cache's memory is uncounted and competes with the arena and the streamed KV |
| The table on the cards | 28.8 GB of VRAM the experts need more; each token needs only 16 rows |

## Consequences

- A slow read lands on the token path until the device waits in the window. Swap on the same SSD, its garbage
  collection and its power states all show up as slow reads.
- With batched windows, a window needs up to 512 rows. That needs more reads in flight: more threads, or io_uring if
  the engine's container allows it.
- The engine keeps the SSD out of deep power states during a run, as Strata does with a keep-alive read.
- The model files live on a bind mount of an ext4 or xfs disk, not inside the container's filesystem.
