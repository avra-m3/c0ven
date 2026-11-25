# ADR-0082: Checkpoints go to disk instead of being dropped

**Status:** draft. 2026-11-24.
**Revisit when:** a served replay shows the writes beside running requests cost more decode time than the prompt
reads they save, or the KV's bytes per token change enough to move the shortest checkpoint worth keeping.

## Context

- A checkpoint is a copy of a slot's running state, about 118 MB of host memory, and a hold on the KV pages before
  it ([CACHING.md](../CACHING.md)). The server drops one when the KV pool is short of pages
  ([ADR-0076](0076-an-admission-evicts-only-checkpoints-that-free-pages.md)) or when the checkpoint pool is full
  ([ADR-0051](0051-checkpoints-stay-spread-back-from-each-conversations-end.md)). A sleep and an exit drop them
  all. Its conversation then reads its whole prompt again: a 100K-token prompt is about a minute of reading
  (*estimate*: 102,400 tokens at the 1,713 tokens a second
  [measured](../measurements/MEASUREMENTS.md#2026-10-20-kv-pool-and-expert-cache-share) for a 179,969-token prompt).
- The KV pages hold 64,896 bytes per four-token page (16,224 bytes a token: K, V, and raw and pooled indexer keys),
  so a 100K-token checkpoint is about 1.8 GB of KV and its state (*estimate*: 102,400 tokens at 16,224 bytes a token
  plus 118 MB). A drive that reads 2 GB a second reads it back in about a second (*estimate*).
- The running state cannot be wound back: a checkpoint serves its own length only, never a prefix of it. A shorter
  checkpoint of the same conversation shares every KV page of its prefix with the longer one.
- Strata's public successor, [mw00/project-maya](https://github.com/mw00/project-maya) at `cfc825f`, keeps
  conversations on an NVMe drive (`include/maya/core/glm_prompt_cache.hpp`): one file each, named by a hash of its
  tokens, found by a chain of hashes over blocks of 256 tokens, removed least recently used first under a cap in
  bytes, and kept across restarts. Its GLM model has no running state, so one file there serves every prefix of
  it. llama.cpp's server saves and restores a slot's whole state to a file on request.

## Decision

1. With `--kv-disk DIR`, a checkpoint the server would drop is written to disk first: when the KV pool needs its
   pages, when the checkpoint pool needs its place and no longer checkpoint in the pool or the incoming one extends it
   (`spread_victim`, [ADR-0051](0051-checkpoints-stay-spread-back-from-each-conversations-end.md)), and before a sleep
   or an exit. It is then dropped as before.
2. An entry on disk is one checkpoint: a file of its running state and tokens, the KV of each whole block of 1,024
   tokens in a block file, and the pages after its last whole block in a tail file. A block file is named by a hash
   of every token up to the block's end, so the checkpoints of one conversation share the block files of the
   prefix they share, and a block is written once.
3. A request whose prompt starts with an entry longer than any checkpoint in memory reads it back before its
   admission once the whole request fits. The entry takes a checkpoint and fresh pages for its unshared blocks, shares
   any whole blocks already held by an in-memory checkpoint, and makes room for the whole request; the admission
   restores it as any checkpoint.
4. While no request runs, the server writes one checkpoint not yet on disk at a time, used longest ago first, so
   an eviction under load finds it there and writes nothing.
5. The folder holds `--kv-disk-gib` GiB at most, 64 by default; the entries used longest ago go first. A file is
   written under a temporary name and renamed, and kept only once it is on the drive. The tier's files bypass the
   page cache.
6. Checkpoints shorter than 1,024 tokens, and those with an image in their prefix, stay off disk.
7. Entries outlive the process. They sit in a subfolder named by the model file, the engine's version, the number of
   cards and `--context`, and each file starts with the layout and `--context` it was written for; the core refuses a
   file of another layout or context, and the entry is then removed.
8. The core writes and reads the files: `engine_save_pages`, `engine_load_pages`, `engine_save_checkpoint` and
   `engine_load_checkpoint`. Python names the files and decides what goes where. ABI version 24.

## Why

The tier costs nothing until a checkpoint is dropped, and a dropped checkpoint read back costs the drive's time for
its bytes in place of the prompt's reading time, about sixty times less for a 100K-token prompt (*estimate*, from
the context's figures). Writing in idle time keeps the writes away from running requests in the common case: a
conversation's checkpoints are taken during its turn and dropped during a later one.

## Options not chosen

- **One file a checkpoint, with all of its KV.** Each of a conversation's checkpoints, 16K tokens apart, would
  write the whole prefix again: eight checkpoints of one 128K-token conversation would write about 10 GB of KV
  in place of 2 (*estimate*).
- **Pages to disk one at a time, as they go stale.** A page alone is of no use: the running state at a
  checkpoint's position is needed to go on from it, so a checkpoint is the unit that goes and comes back.
- **Writes through the page cache.** Faster to return, but gigabytes of written pages would compete with the arena
  and the n-gram table for the host's memory.
- **Writes on a thread of their own.** The pages' bytes come off the cards through the copy lists, which only one
  thread may use ([ADR-0077](0077-one-thread-uses-the-cards.md)).

## Consequences

- An eviction under load whose checkpoint is not on disk yet waits for its write, and the cards for it: the plans
  queued before it finish, and the next waits. The writes of item 4 make this the rare case.
- A read back holds the cards in the same way while it runs.
- One server uses a folder at a time; two would remove each other's files.
- The STATS line counts the tier's entries, bytes, writes, reads and their time.
