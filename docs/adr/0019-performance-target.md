# ADR-0019: performance - a target set by the owner

**Status:** accepted, 2026-10-10, by the owner. Supersedes [ADR-0013](0013-performance-floor.md).
**Revisit when:** the owner changes the target, or P50's forecast shows it out of
reach.

## Context

ADR-0013 defined success as beating llama.cpp on this machine "clearly". That needed three baseline plans and two
"bars" (P20-P22, P35, P36) before the goal itself was known. The owner gives the goal directly instead.

## Decision

1. **The target, for one request on both B60s, with the IQ3_S file ([ADR-0018](0018-one-file-iq3s.md)):**

   | | Target |
   | --- | ---: |
   | Decode, one request, no speculative decoding (no draft layer, one token per step) | **30 tokens/s** |
   | Prompt | **2,000 tokens/s** |

   The owner did not name a context length. Both are measured at 4K and at 128K; the target applies at 4K until the
   owner says otherwise.
2. **The go/no-go** (P50) forecasts both numbers from the plans' measurements. If
   the forecast is well short, work stops and the owner decides before any engine code.
3. **The stop rule:** at stage 5, the fast decode loop (PLAN.md), decode with
   drafting off is measured against 30 tokens/s. If it is well short, work stops and the plan is reconsidered.
   Prompt speed is checked at stage 6.
4. **No llama.cpp baseline is a gate.** A `llama bench` run on the IQ3_S file may be recorded once for comparison; it
   decides nothing.
5. **Several requests at once** keep [ADR-0006](0006-scope.md)'s capability target (4 at full context). No total
   tokens/s target is set for them.

## Consequences

- P20, P21, P22 and P35 are dropped. P36 stays, as the CPU's share of the experts is a large part of decode with
  IQ3_S.
- ADR-0018's consequence about the llama.cpp floor needing every type on SYCL no longer applies.
- Every speed claim names the target, the stage, and the context length it was measured at.
