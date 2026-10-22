# ADR-0009: two cards - each owns a range of layers, and only the hand-off crosses, through host memory

**Status:** superseded by [ADR-0023](0023-two-cards-tensor-split.md) on 2026-10-12. Was accepted on 2026-10-09.
**Revisit when:** superseded.

## Context

- **The two B60s share no link.** Each has its own PCIe 5.0 x8 link to the CPU, so anything that goes from one card
  to the other crosses PCIe twice.
- **The cost of crossing is the number of synchronisations, not the bytes.** llama.cpp's tensor split on two B70s
  (two exchanges per layer) decoded a 27B model at 17.7 tokens/s. One B70 alone ran a similar 27B model at 22.9.
  The models and quants differ, so the comparison is rough, but the split did not pay.
- **Pinned host buffers are the fastest exchange path on Intel**
  ([llama.cpp #29604](https://github.com/ggml-org/llama.cpp/pull/29604)).

## Decision

1. **Split by layers.** Each card owns a contiguous range of layers and everything they touch: their dense weights,
   expert cache, KV and recurrent state, and prompt buffers. Nothing a card owns is copied to the other card.
2. **Only the residual hand-off crosses,** once per window, through pinned host memory. No peer-to-peer.
3. **No tensor or expert parallelism.**
4. **Memory:**
   - one Level Zero context holds both cards;
   - device memory for what a card owns, host memory for what both cards or the CPU touch, and nothing that migrates
     pages behind the engine's back;
   - large blocks are allocated per layer.
5. **Per card:** an in-order compute queue, an in-order copy queue, and its own host thread.
6. **The two stages are chained on the devices.** Card 0 raises a flag ([ADR-0011](0011-cpu-gpu-signalling.md)), and
   card 1 waits on it. The host joins only at the end of the window.
7. **The split point balances time per window,** not only expert-cache coverage, once requests are pipelined
   ([ADR-0010](0010-several-requests.md)).

## Consequences

- **RAM bandwidth is the shared limit,** not PCIe: the CPU, both cards' streamed misses and the far KV all read the
  same RAM.
- **One request uses one card at a time.** ADR-0010 recovers the idle half.
