# ADR-0080: two-card exchanges fall back to pinned host memory when peer access is unavailable

**Status:** draft, 2026-11-21. Extends [ADR-0030](0030-exchanges-into-the-other-card.md).
**Revisit when:** a host without peer access shows a faster host-staged exchange path, or a third card layout arrives.

## Context

- [ADR-0030](0030-exchanges-into-the-other-card.md) writes two-card exchanges straight into the other card's VRAM: the
  decode step's partial sums from two rows up, the prompt path's partial sums, indexer candidates and split
  hyper-connections, and the image encoder's K and V.
- Peer access is not available on every two-card machine. Linux's `pci_p2pdma` allows cross-root-port transfers on AMD
  Zen host bridges, and on a fixed list of Xeon server bridges, but refuses consumer Intel host bridges and PCIe
  switches whose ACS settings route peer traffic back through the host bridge.
- On a refused path, `zeCommandListAppendMemoryCopy` between two cards returns `0x70000003`
  (`ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) while recording the decode step's first two-row list, stopping the load even
  with free VRAM.
- `zeDeviceCanAccessPeer` alone is not enough to trust the path: a bridge or switch that redirects peer writes without a
  kernel refusal can return `ZE_RESULT_SUCCESS` and leave wrong bytes in the destination buffer. Testing only a copy
  also misses a host whose copy path works while a kernel's direct stores into the other card do not.

## Decision

1. `ze::Runtime::peer_access` probes peer access at load. It checks `zeDeviceCanAccessPeer` and
   `zeDeviceGetP2PProperties` (`ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS`) in both directions, copies a 4 KiB pattern each way
   on a regular command list and checks every byte read back, and runs `smoke` (`kernels/smoke.cl`, 64-byte stores per
   sub-group) from each card into the other card's memory and checks every 32-bit value read back. Any refusal, error or
   mismatch turns peer access off and logs the reason.
2. Setting `ENGINE_NO_P2P=1` disables peer access on a two-card host so the fallback can be tested and timed on machines
   whose peer path works.
3. Without peer access, two-card exchanges use pinned host memory (`zeMemAllocHost`):
   - Decode steps of two rows or more stay on `ex1_` and `ex2_` (already allocated for one-row steps) and skip
     `Card::xp`.
   - Prompt partial sums write via the existing 64-byte kernel stores into a double-buffered host array (`pex_`,
     `[2][card][tmax_][embd]`, allocated once at load so it survives sleep), and `p_hc_combine` / `p_hc_combine_norm`
     read the other card's slot from it. `PromptCard::recv` is skipped.
   - Prompt indexer candidates copy `P.cand` into a host buffer (`picand_`, `[card][tmax_][idx_cand()]`, allocated once
     at load) and `p_idx_merge` reads the other card's slice from it. One slot per card suffices because the layer's
     mixer exchange synchronises both cards before the next attention layer. `PromptCard::cand_peer` is skipped.
   - Prompt hyper-connection row splitting (`kHcSplitRows`) stays off, so both cards compute every row and neither
     `p_hc_mix` nor `p_hc_share` writes peer memory.
   - The image encoder copies each card's K and V rows into pinned host staging (`vk_host_`, `vv_host_`, allocated once
     in `load_vision`), waits for the other card, and copies the other card's rows into its own `V.k` and `V.v` before
     `v_attn`.
4. When peer access works, recorded lists, buffers and kernel arguments stay identical to
   [ADR-0030](0030-exchanges-into-the-other-card.md).

## Options not chosen

- Trust `zeDeviceCanAccessPeer` without live transfers, or test copies without kernel stores. Either leaves silent
  corruption uncaught when a bridge drops peer writes or a driver handles copies and kernel stores on separate paths.
- Host-stage the split hyper-connections' `x16` and `xb`. Moving `2 * 2,176 * 2,560 * 2` = 22.3 MB per card twice per
  layer across 47 layers through DDR5 (computed at `tmax_ = 4,352`, `embd = 2,560`) would contend with expert staging
  and cost more than computing the rows locally on each card
  ([the split record](../measurements/MEASUREMENTS.md#2026-10-17-hc-split)).
- Copy `P.part` into host memory and back into `P.recv`. Moves the full bucket width `T` instead of actual rows `*ntok`
  and doubles PCIe transfers per exchange.
- Run the image encoder on one card without peer access. Halving the card count shrinks `vmax_tokens_` in `carve`,
  reducing the largest image the prompt regions hold.

## Consequences

- Two-card hosts without PCIe P2P load and run instead of failing at `open` with `0x70000003`.
- Without peer access, pinned host memory grows by 119.1 MiB without vision and up to 199.1 MiB with vision, while each
  card frees 60.2 MiB of VRAM (computed at `tmax_ = 4,352`, `embd = 2,560`, `idx_cand = 1,028`, `vmax_tokens_ <= 4,096`:
  `pex_` is `2 * 2 * 4,352 * 2,560 * 2` = 85.0 MiB, `picand_` is `2 * 4,352 * 1,028 * 4` = 34.1 MiB, `vk_host_` and
  `vv_host_` are `2 * 4 * 4,096 * 1,280 * 2` <= 80.0 MiB; skipped per card are `Card::xp` 0.6 MiB, `PromptCard::recv`
  42.5 MiB and `PromptCard::cand_peer` 17.1 MiB).
- Expected throughput cost when peer access is off (*estimate*, reversing the gains measured in
  [MEASUREMENTS.md](../measurements/MEASUREMENTS.md)): 1-2% on multi-row decode
  ([decode peer exchange](../measurements/MEASUREMENTS.md#2026-10-16-peer-exchange-decode)), and 8-10% on long prompt
  chunks combining the host-exchange baseline of `p_hc_combine_norm` (2,353.5 against 2,427 tokens/s in the
  [prompt exchange record](../measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores); direct kernel writes
  into host memory are unmeasured until run on hardware) and unsplit hyper-connections
  ([HC split](../measurements/MEASUREMENTS.md#2026-10-17-hc-split)), plus host reads of indexer candidates past 2,048
  tokens.
- Zero runtime cost on machines with working peer access beyond the startup probe.
