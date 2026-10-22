# ADR-0030: the cards' exchanges may write into the other card's memory

**Status:** accepted, 2026-10-16. Supersedes item 2 of
[ADR-0023](0023-two-cards-tensor-split.md) in part ("No peer-to-peer").
**Revisit when:** a driver or kernel update changes peer access (`zeDeviceGetP2PProperties`), or the exchanges move to
flags spun on by the cards.

## Context

- **ADR-0023, item 2,** combines the cards' halves by exchanges through host memory, and rules out peer-to-peer.
- **Peer access works.** `zeDeviceCanAccessPeer` is true both ways, `zeDeviceGetP2PProperties` reports access without
  atomics, and a kernel's writes into the other card read back exact
  ([the peer exchange record](../measurements/MEASUREMENTS.md#2026-10-16-peer-exchange)).
- **Decode's exchange is cheaper through the other card at two rows and more.** In the engine's form (an event per
  exchange), 2,560 floats a row both ways cost 10.2-10.3 µs through host memory at four rows and 5.1-5.2 µs written
  by the kernel into the other card: 0.48 ms saved a step of 96. At one row 3.9-5.0 against 2.3-3.4 µs, 0.15 ms a step
  ([the same record](../measurements/MEASUREMENTS.md#2026-10-16-peer-exchange)).
- **The prompt path already does it.** Its kernels write their partial sums into the other card, which took its
  prompt from 2,353.5 to 2,427 tokens/s ([record](../measurements/MEASUREMENTS.md#2026-10-15-prompt-exchange-stores)).
  That departs from item 2 as written.
- **The host memory link is shared.** The exchanges through host memory compete with the CPU workers' reads of the
  arena and the far KV ([ADR-0023](0023-two-cards-tensor-split.md), consequences). Writes into the other card do not
  touch host memory.

## Decision

1. **A card may write data it sends the other card straight into that card's memory,** by a kernel's stores of at
   least 64 bytes or by a copy, in place of host memory. Each change that does is judged as any other
   ([ADR-0016](0016-how-a-change-is-judged.md)): ABBA, fresh processes, every length measured.
2. **The signal stays an event** (or a flag in host memory, [ADR-0011](0011-cpu-gpu-signalling.md)). No atomics across
   the cards: the driver reports none.
3. **What the CPU reads stays in host memory:** the CPU's answers and anything the host thread reads.
4. **The rest of ADR-0023 stands:** every layer split across both cards, the CPU computing the misses, one context,
   device memory for what a card owns.

## Why

Half the exchange's cost at four rows, with every value checked, and no change to the arithmetic: the same partial
sums are added in the same order.

## Options not chosen

- **Keep item 2 as it is.** Leaves about 0.27 ms a decode plan (*estimate*, from the step mix at long-4k) and keeps the
  prompt path out of line with the ADR.
- **Peer writes for one row too.** Allowed by item 1, but at 0.15 ms a step it is the smaller part; the first change
  to try is windows of two rows or more.

## Consequences

- The prompt path's peer writes are covered.
- The first decode change to try is the out products and the MoE down product writing into the other card for
  windows of two rows or more, timed at long-4k, 32k, 128k and 256k.
- [DESIGN.md](../DESIGN.md) changes with the engine, in the commit that moves an exchange.
