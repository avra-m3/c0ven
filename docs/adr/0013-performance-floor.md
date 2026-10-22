# ADR-0013: performance - llama.cpp on this machine is the floor

**Status:** superseded by [ADR-0019](0019-performance-target.md) on 2026-10-10, which sets the target directly. Was
accepted on 2026-10-09.
**Revisit when:** superseded.

## Context

llama.cpp already runs this model on these cards. A new engine is only worth its cost if it is clearly faster on the
same work.

## Decision

1. **The floor is llama.cpp on this machine with the engine's files and settings:**
   - both B60s, a layer split, 8-bit KV;
   - the experts that do not fit on the cards, on the CPU;
   - decode and prompt speed at 4K and 128K;
   - with several requests at once.
2. **The stop rule:** the engine must beat the single-request floor clearly once the fast decode loop is built. If it
   does not, work stops and the plan is reconsidered.
3. **Serving several requests must beat llama.cpp's numbers with several requests.**
4. **A llama.cpp setup with Unsloth's file and f16 KV** is recorded for comparison, but is not the floor.

## Consequences

- The floor is measured before the engine exists.
- Every speed claim names the floor it is compared with.
