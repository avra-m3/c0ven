# ADR-0035: the arena in the engine's own memory, read in place by the copy engines

**Status:** accepted, 2026-10-17; item 2 and the context amended 2026-10-18 to name the mechanism
[the swaps record](../measurements/MEASUREMENTS.md#2026-10-17-swaps-keep-the-copies) measured. Supersedes item 6 of
[ADR-0023](0023-two-cards-tensor-split.md) in part: the arena is no longer `zeMemAllocHost` memory.
**Revisit when:** a driver or kernel update changes how the runtime copies from host memory it did not allocate
([LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md) row 31); xe fails in its userptr path; or a host whose cgroup has swap swaps the
arena out.

## Context

- **The driver's arena stalls decode.** When the host reclaims, xe moves its host buffers out of GTT and drops the
  CPU's mappings of them, and the arena is most of those buffers. One page-cache read of the model file stalled
  decode for up to 5.75 s
  ([eviction record](../measurements/MEASUREMENTS.md#2026-10-17-decode-stalls-eviction),
  [under-reclaim record](../measurements/MEASUREMENTS.md#2026-10-17-anon-arena-under-reclaim)).
- **xe cannot move anonymous memory.** With the arena in anonymous `mmap` pieces, the same read left the arena's
  11.61 GB in place, and no decode step took over 48.1 ms (under-reclaim record).
- **Both cards read it in place.** Their copy engines read anonymous memory correctly at 28.4 GB/s, the rate from
  `zeMemAllocHost`, and no page moved (the same record). compute-runtime binds each copy's source range to the card
  as a userptr when a copy list is recorded, 21-46 µs a copy, and unbinds it when the list is freed, 50-89 µs. It
  does not use shared system USM for these copies, though each card reports it (`sharedSystemAllocCapabilities`
  `0xf`) ([the swaps record](../measurements/MEASUREMENTS.md#2026-10-17-swaps-keep-the-copies)).
- **Each copy from it costs more.** A copy of 330 KB took about 1.3 µs longer from anonymous memory, with huge pages
  or without. Over the prompt path's 17,712 copies a card a chunk, 64-token prompts took 17-21 ms longer. Region
  copies, about 150 a card, remove the difference ([the A/B record](../measurements/MEASUREMENTS.md#2026-10-17-anon-arena-default)).

## Decision

1. **The arena is the engine's own anonymous memory** (`Options::anon_arena`, on by default): 1 GiB pieces from
   `mmap`, 2 MiB aligned, with `MADV_HUGEPAGE`. `false` keeps the driver's memory, for A/B runs.
2. **The copy engines read it in place,** through the runtime's binding of each copy's source: no import and no
   second copy. The prompt path records its copy list once, at load, and a swap keeps every staging slot's arena
   piece, so no binding is made or undone after load.
3. **The prompt path stages each run of experts adjacent in the arena with one region copy a matrix**
   (`ze::List::copy_rows`), whichever memory the arena is.
4. **The rest of ADR-0023 item 6 stands:** one context, device memory for what a card owns, `zeMemAllocHost` for the
   other host memory both cards or the CPU touch, and never `zeMemAllocShared`.

## Why

It removes the stalls and costs nothing measured. Over three ABBA rounds against the driver arena with per-expert
copies, the tokens were the same, 64-token prompts 4% faster, and longer prompts, decode and loads as fast (the A/B
record).

## Options not chosen

- **Importing the arena with the memmap extension.** One context's import read correctly on card 0 and reset card
  1's engine ([host-memory record](../measurements/MEASUREMENTS.md#2026-10-12-host-memory), Q6). Reading in place needs no
  import.
- **Two copies, one for the CPU and one for the copy engines.** It doubles the arena, and the driver can still move
  the second copy.
- **Keeping the driver's buffers from moving.** xe has no call that pins a buffer from user space.

## Consequences

- The arena, 10.85-20.5 GiB, counts against the engine's cgroup. A serving container's memory limit must allow it.
- Where the cgroup has swap, the host may swap out experts left unread for minutes. Reading one back costs
  milliseconds, not seconds (*estimate*: the SSD's time for 2 MB). Not measured.
- The copy engines read through userptr bindings, which the kernel invalidates when it moves or reclaims the
  arena's pages. Prompts under the host's reclaim with this arena are not measured. xe can fail under heavy host
  memory pressure ([stalls record](../measurements/MEASUREMENTS.md#2026-10-14-decode-stalls)); it did not fail in the
  runs above, one of them under the host's reclaim.
- Asleep, the engine keeps the arena as [ADR-0032](0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md)
  item 2 says; it is now the child's own memory.
- [DESIGN.md, section 6](../DESIGN.md#6-memory) describes the arena as this ADR has it.
