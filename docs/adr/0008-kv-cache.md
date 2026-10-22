# ADR-0008: an 8-bit KV cache at full context, streamed to RAM

**Status:** accepted, 2026-10-09.
**Revisit when:** host memory cannot be pinned at the sizes KV streaming needs
([host memory record](../measurements/MEASUREMENTS.md#2026-10-12-host-memory)). The fallback would shorten the
context, which conflicts with decision 2.

## Context

A sequence's KV cache at 262K tokens is ~3.6 GB at 8 bits (*estimate*, from the model's shapes), and the target is
4 such sequences. VRAM is better spent on experts. The RAM has room.

## Decision

1. **The KV cache is 8-bit.** No KV format below 8 bits: below 8 bits causes real quality problems with Qwen models.
   Strata's 4-bit KV is not rebuilt.
2. **The context is never shortened** to make room for experts.
3. **Full context for several requests fits through KV streaming:**
   - Only the most-read cells of each attention layer stay in VRAM (32,768); the rest stay in host memory.
   - The attention kernels read both, and the attention reads the same values as before.
4. **A fixed KV and state reserve per card** is set aside before the expert cache is sized.

## Consequences

- KV streaming needs pinned host memory read by kernels over PCIe.
- Streamed KV takes ~13 GB of RAM for 4 sequences at 262K (*estimate*; [MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md)).
- The 8-bit KV kernels and the streaming are among the first attention kernels.
