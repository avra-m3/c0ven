# ADR-0013: performance - llama.cpp on this machine is the floor

**Status:** superseded by [ADR-0019](0019-performance-target.md) on 2026-10-10: the owner sets the target
directly. Was accepted, from INTEL_SYCL_PLAN.md stage -1 ("the floor the port must beat ... If [it] does not beat it
clearly, stop and reconsider"), merged in PR #1 (2026-10-09).
**Revisit when:** llama.cpp improves. The floor is re-measured with each llama.cpp upgrade on the machine.

## Context

llama.cpp already runs this model on these cards. A new engine is only worth its cost if it is clearly faster
on the same work.

## Decision

1. **The floor is llama.cpp on this machine with the engine's files and settings:**
   - both B60s, a layer split, 8-bit KV;
   - the experts that do not fit on the cards, on the CPU;
   - decode and prompt speed at 4K and 128K ([P21](../precursor/P21-baseline-floor.md));
   - with several requests at once ([P22](../precursor/P22-baseline-concurrency.md)).
2. **The stop rule:** the engine must beat the single-request floor clearly at its first fast stage. If it does not,
   work stops and the plan is reconsidered before going further. That stage is stage 5, the fast decode loop
   ([PLAN.md](../PLAN.md#the-engines-stages)). [P46](../precursor/P46-stages-and-readiness.md) says what "clearly"
   means in numbers.
3. **The concurrency stages must beat P22's numbers.**
4. **Today's deployed setup** (Unsloth's file, f16 KV; [P20](../precursor/P20-baseline-as-deployed.md)) is recorded for
   comparison, but is not the floor.

## Consequences

- The floor is measured before the engine exists.
- Every speed claim names the floor it is compared with.
