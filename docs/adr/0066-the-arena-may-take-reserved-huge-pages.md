# ADR-0066: the arena may take the host's reserved huge pages

**Status:** draft. 2026-11-07.
**Revisit when:** a run with `--hugepages` is timed against one without it, or the kernel starts moving reserved huge
pages under the arena's bindings.

## Context

- The cards' copy engines read the expert arena in place, through a binding of each copy's source pages
  ([ADR-0035](0035-the-arena-in-the-engines-own-memory.md)). When the kernel moves a small arena page, the card stops
  while the binding is made again.
- The arena takes transparent huge pages only while the host keeps free 2 MiB blocks (`kKeepFreeHuge` in the core),
  and `MADV_COLLAPSE` at load is best effort. Pieces past that limit, and ranges that do not collapse, stay in small
  pages, which compaction may move.
- The other host memory the cards read comes from `zeMemAllocHost`. The driver pins it and nothing binds it, so it
  has no such stall.
- Reserved huge pages (`vm.nr_hugepages`) are never swapped, compacted or split. A private `MAP_HUGETLB` mapping
  reserves its pages when it is made, so a short pool shows at `mmap`, not at a later fault.

## Decision

1. `--hugepages` (engine flag; `ENGINE_CONFIG_HUGETLB`, ABI version 19) takes each arena piece from the reserved huge
   pages of the default size, while they last. A piece they cannot hold is made as without the flag, and the log
   says so once.
2. A piece in reserved huge pages is not locked and not collapsed.
3. Only the arena takes them. The pinned buffers stay with the driver.
4. The memory plan counts the arena's part in reserved huge pages apart from `MemAvailable` and `memory.max`, which
   leave them out, and warns when the pool holds less than the arena.
5. The flag is off by default.

## Options not chosen

- **Refusing to load when the pool is short.** A swap round can grow the arena after the load, and failing there
  would stop a running server.
- **Reserved huge pages for the pinned buffers.** The driver allocates them and nothing moves them under a binding.

## Consequences

- An operator reserves the pages on the host before the start, for example `sysctl vm.nr_hugepages=<arena GiB x
  512>` for 2 MiB pages. They stay out of the host's other use while reserved.
- No speed is claimed for the flag until a timed run measures it.
