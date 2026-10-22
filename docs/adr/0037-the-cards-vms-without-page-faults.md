# ADR-0037: the cards' VMs without page faults

**Status:** draft, 2026-10-19; not accepted. Would supersede in part [ADR-0002](0002-pin-the-toolchain.md)'s
row "environment the engine sets" (it adds a key) and the third consequence of
[ADR-0035](0035-the-arena-in-the-engines-own-memory.md) (xe's failure in its userptr path).
**Revisit when:** a compute-runtime update renames or drops `EnableRecoverablePageFaults`, or makes it a release
setting; xe retries a userptr re-pin that fails; or the engine needs shared system USM.

## Context

- **compute-runtime creates each card's VM in page-fault mode by default.** On a B60 with the pinned stack, the
  VM is created with long-running mode and `DRM_XE_VM_CREATE_FLAG_FAULT_MODE`, and each card reports shared
  system USM (`sharedSystemAllocCapabilities` `0xf`)
  ([the fault-mode record](../measurements/MEASUREMENTS.md#2026-10-19-vm-fault-mode)).
- **The copy engines read the anonymous arena through userptr bindings** of each copy's source
  ([ADR-0035](0035-the-arena-in-the-engines-own-memory.md) item 2; [LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md) row 31).
  xe does not pin their pages. When the kernel migrates a page of a bound range, as compaction does with 4 KiB
  pages, xe invalidates the binding. In fault mode, the card's next access faults, and xe re-pins the whole range.
- **In fault mode, a migration during the copies resets the copy engine.** The fault handler does not retry a
  re-pin that fails, so the card's page fault fails and the GuC resets the engine. A 512 MiB copy loop and the
  engine's prompts both reset a copy engine, on either card, and the engine then fails its next list wait ([the fault-mode record](../measurements/MEASUREMENTS.md#2026-10-19-vm-fault-mode)).
- **Without fault mode nothing reset.** With `NEOReadDebugKeys=1` and `NEO_EnableRecoverablePageFaults=0`, the VM is
  created in long-running mode only, and the cards report no shared system USM (`0x0`). By xe's source, a migrated
  page then stops the VM's queues until xe binds the range again. The same copy loop and the engine on two cards ran
  clean under tens of thousands to millions of host migrations, with the same tokens and prompt speed (the same
  record). In three ABBA rounds on two cards, the mode changes neither prompt reading, nor decode, nor the first token
  ([the speed record](../measurements/MEASUREMENTS.md#2026-10-19-vm-without-fault-mode-speed)). The setting alone does
  nothing: a release build reads it only with `NEOReadDebugKeys=1`.
- **The engine uses no shared system USM.** It never passes the cards a host pointer it did not bind through a
  copy, and the runtime binds a copy's source with or without shared system USM.

## Decision

1. **The engine creates the cards' VMs without page faults.** Before the process's first `zeInit`, `ze::init` sets
   `NEOReadDebugKeys=1` and `NEO_EnableRecoverablePageFaults=0`, unless the environment already sets them. The
   core's `ze::Runtime` and its tests' card count call `zeInit` through it.
2. **The anonymous arena does not need shared system USM.** The core keeps `anon_arena` whatever the cards report,
   and logs each card's capabilities at load.

## Options not chosen

- **Bind only ranges in 2 MiB pages.** Compaction does not migrate huge pages, so no binding would be invalidated.
  It makes full huge-page coverage a condition of serving an expert from the copy engines, which a fragmented host
  cannot meet. Not measured beyond one clean run.
- **`NEO_EnableSharedSystemUsmSupport=0`.** It leaves fault mode as well (the same record). It names a feature
  rather than the mode, and turning off page faults is the change that matters.
- **The driver's memory for the arena.** It brings back the decode stalls of ADR-0035.
- **Per-expert copies,** to make each re-pin short. A shorter walk fails less often, not never, and it gives back
  what region copies saved.

## Consequences

- A host migration under a binding costs the card a stall instead of a reset. The copy loop's slowest copy rose
  from about 0.02 s to 0.076-0.127 s (the same record). Under the engine's prompts and decode, with the arena in
  2 MiB pages, no cost was measured ([the speed record](../measurements/MEASUREMENTS.md#2026-10-19-vm-without-fault-mode-speed)).
- `NEOReadDebugKeys=1` makes the runtime read every compute-runtime debug key from the environment, so a stray key
  in a serving container takes effect.
- [LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md) row 31 says how the copies from the arena behave in each mode.
