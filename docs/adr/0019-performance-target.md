# ADR-0019: performance - a set target

**Status:** accepted, 2026-10-10. Supersedes [ADR-0013](0013-performance-floor.md). Superseded by
[ADR-0056](0056-the-aim-is-the-fastest-performance-on-battlemage.md) (draft).
**Revisit when:** the target changes, or the forecast built on the
[design-inputs record](../measurements/MEASUREMENTS.md#2026-10-13-design-inputs) shows it out of reach.

## Context

ADR-0013 defined success as beating llama.cpp on this machine "clearly". That needed several baseline measurements
before the goal itself was known. A target set directly removes them.

## Decision

1. **The target, for one request on both B60s, with the IQ3_S file ([ADR-0018](0018-one-file-iq3s.md)):**

   | | Target |
   | --- | ---: |
   | Decode, one request, no speculative decoding (no draft layer, one token per step) | **30 tokens/s** |
   | Prompt | **2,000 tokens/s** |

   Both are measured at 4K and at 128K; the target applies at 4K.
2. **The go/no-go** forecasts both numbers from measurements before any engine code. If the forecast is well short,
   work stops.
3. **The stop rule:** once the fast decode loop is built, decode with drafting off is measured against 30 tokens/s.
   If it is well short, work stops and the plan is reconsidered. Prompt speed is checked once the prompt path is
   built.
4. **No llama.cpp baseline is a gate.** A `llama bench` run on the IQ3_S file may be recorded for comparison; it
   decides nothing.
5. **Several requests at once** keep [ADR-0006](0006-scope.md)'s capability target (4 at full context). No total
   tokens/s target is set for them.

## Consequences

- The CPU's share of the experts stays a large part of decode with IQ3_S, so its speed is measured.
- ADR-0018's need for every type on llama.cpp's SYCL backend no longer applies.
- Every speed claim names the target and the context length it was measured at.
