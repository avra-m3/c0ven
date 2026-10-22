# ADR-0011: the CPU and GPU signal through flags in host memory; spinning waits first

**Status:** accepted, 2026-10-09. Confirmed by the doorbell probe on 2026-10-11
([record](../measurements/MEASUREMENTS.md#2026-10-11-doorbell)).
**Revisit when:** spinning on host memory proves unreliable on Xe2 (stale values, hangs, job timeouts), or slower
than events. Or when the skip stays rare even with prefetching
([hit rates](../measurements/MEASUREMENTS.md#2026-10-12-expert-hit-rates),
[prefetch](../measurements/MEASUREMENTS.md#2026-10-14-expert-prefetch)), and an event wait costs no more than a
spin.

## Context

In every layer of a window:

1. The GPU publishes the experts it chose.
2. The CPU computes the ones not resident on the card.
3. The GPU waits for the CPU's results.

Strata does this with no driver calls on the token path. A Level Zero event wait recorded in a command list cannot
be skipped. A wait kernel can be skipped when the device planned the layer itself, because all its experts are
resident (Strata's E-6).

The skip needs every one of a window's ~25 distinct experts in the layer to be resident. Even at 85% hits that is a
minority of layers (*estimate*: 0.85^25 ≈ 2% if experts were independent, a few times that with correlated
routing). Prefetching a layer's few misses is what can make the skip common: once all of a layer's misses have
landed in the prefetch ring, the device can plan the layer itself. So flags matter first for having no driver calls
on the token path, and later for the skip.

## Decision

1. **Signals are flags in host memory.** No driver calls on the token path.
2. **Wait kernels spin on those flags with system-scope acquire loads.** Writers publish with a fence and a
   system-scope release store.
3. **A wait returns at once when the device planned the layer itself** (the skip).
4. **Level Zero events are the measured fallback:** signalled by the host, waited on in the command list.
5. **The last resort:** the host enqueues the next layer's expert work after reading the ids.
6. **Every spin is bounded and lives within one window,** never across idle time. A watchdog releases spinning waits
   before the engine stops, within the `xe` job timeout.
7. **The same flag pattern chains the two cards** ([ADR-0009](0009-two-cards.md)).

## Consequences

- **Correctness depends on memory ordering across PCIe** on Xe2. It is tested in both directions, requiring 100 of 100
  rounds to be seen, before the engine relies on it, and again in the engine's own tests.
- **If spinning fails,** every layer pays a CPU round trip, and the decode-speed estimates drop.
