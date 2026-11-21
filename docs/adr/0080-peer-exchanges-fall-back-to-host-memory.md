# ADR-0080: two-card exchanges fall back to pinned host memory when peer access is unavailable

**Status:** draft, 2026-11-21. Extends [ADR-0030](0030-exchanges-into-the-other-card.md).
**Revisit when:** a host without peer access shows a faster host-staged exchange path, or a third card layout arrives.

## Context

- [ADR-0030](0030-exchanges-into-the-other-card.md) writes two-card exchanges straight into the other card's VRAM: the
  decode step's partial sums from two rows up, the prompt path's partial sums, indexer candidates and split
  hyper-connections, and the image encoder's K and V.
- Peer access is not available on every two-card machine. Linux's `pci_p2pdma` allows cross-root-port transfers on AMD
  Zen host bridges, and on a fixed list of Xeon server bridges, but refuses consumer Intel host bridges and PCIe
  switches whose downstream ports have ACS redirect enabled.
- On a refused path, `zeCommandListAppendMemoryCopy` between two cards returns `0x70000003`
  (`ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY`) while recording the decode step's first two-row list, stopping the load even
  with free VRAM.
- `zeDeviceCanAccessPeer` alone is not enough to trust the path: a bridge or switch that redirects peer writes without a
  kernel refusal can return `ZE_RESULT_SUCCESS` and leave corrupt bytes in the destination buffer.

## Decision

1. **Probe peer access at load.** `ze::Runtime::peer_access` checks `zeDeviceCanAccessPeer` and
   `zeDeviceGetP2PProperties` (`ZE_DEVICE_P2P_PROPERTY_FLAG_ACCESS`) in both directions, then copies a 4 KiB pattern
   each way on a regular command list and checks every byte read back. Any refusal, error or mismatch turns peer access
   off and logs the reason.
2. **Allow forcing the fallback.** Setting `ENGINE_NO_P2P=1` disables peer access on a two-card host so the fallback
   can be tested and timed on machines whose peer path works.
3. **Fall back to pinned host memory (`zeMemAllocHost`) without changing the hot path when peer access works:**
   - **Decode steps of two rows or more:** stay on `ex1_` and `ex2_` (already allocated for one-row steps) and skip
     `Card::xp`.
   - **Prompt partial sums:** each card writes its fp16 partial sums via the existing 64-byte kernel stores into a
     double-buffered host array (`pex_`, `[2][card][tmax_][embd]`, allocated once at load so it survives sleep), and
     `p_hc_combine` / `p_hc_combine_norm` read the other card's slot from it. `PromptCard::recv` is skipped.
   - **Prompt indexer candidates:** each card copies `P.cand` into a host buffer (`picand_`, `[card][tmax_][idx_cand()]`,
     allocated once at load) and `p_idx_merge` reads the other card's slice from it. One slot per card suffices because
     the layer's mixer exchange synchronises both cards before the next attention layer. `PromptCard::cand_peer` is
     skipped.
   - **Prompt hyper-connections:** row splitting (`kHcSplitRows`) stays off, so both cards compute every row and neither
     `p_hc_mix` nor `p_hc_share` writes peer memory.
   - **Image encoder K and V:** each card copies its K and V rows into pinned host staging (`vk_host_`, `vv_host_`,
     allocated once in `load_vision`), waits for the other card, and copies the other card's rows into its own `V.k` and
     `V.v` before `v_attn`.
4. **No change when peer access works.** Recorded lists, buffers and kernel arguments stay identical to
   [ADR-0030](0030-exchanges-into-the-other-card.md).

## Options not chosen

- **Trust `zeDeviceCanAccessPeer` without a live copy.** Leaves silent data corruption uncaught on a switch or patched
  kernel whose root complex drops peer TLPs.
- **Host-stage the split hyper-connections' `x16` and `xb`.** Moving 22.3 MB per card twice per layer across 47 layers
  through DDR5 (*estimate*, from `2 * 2,176 * 2,560 * 2` bytes at `tmax_ = 4,352`) would contend with expert staging and
  cost more than computing the rows locally on each card
  ([the split record](../measurements/MEASUREMENTS.md#2026-10-17-hc-split)).
- **Copy `P.part` into host memory and back into `P.recv`.** Moves the full bucket width `T` instead of actual rows
  `*ntok` and doubles PCIe transfers per exchange.
- **Run the image encoder on one card without peer access.** Halving the card count shrinks `vmax_tokens_` in
  `carve`, reducing the largest image the prompt regions hold.

## Consequences

- Two-card hosts without PCIe P2P load and run instead of failing at `open` with `0x70000003`.
- Without peer access, pinned host memory grows by about 119 MiB without vision (`pex_` 85 MiB, `picand_` 34 MiB at
  `tmax_ = 4,352`) and up to 199 MiB with vision (`vk_host_` and `vv_host_` up to 80 MiB). Each card frees about 60 MiB
  of VRAM (`Card::xp`, `PromptCard::recv`, `PromptCard::cand_peer`).
- Expected throughput cost when peer access is off (*estimate*, from the P2P gains measured in
  [MEASUREMENTS.md](../measurements/MEASUREMENTS.md)): about 1-2% on multi-row decode
  ([decode peer exchange](../measurements/MEASUREMENTS.md#2026-10-16-peer-exchange-decode)) and about 8-10% on long
  prompt chunks combining host-memory partial sums ([prompt exchange stores](../measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores))
  and unsplit hyper-connections ([HC split](../measurements/MEASUREMENTS.md#2026-10-17-hc-split)), plus host reads of
  indexer candidates on prompts past 2,048 tokens.
- Zero runtime cost on machines with working peer access beyond the 4 KiB startup probe.
