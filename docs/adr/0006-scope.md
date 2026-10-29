# ADR-0006: scope - one machine, one model family, full context, several requests

**Status:** accepted, 2026-10-09. Item 1 superseded in part (one machine), and item 2, by
[ADR-0056](0056-the-aim-is-the-fastest-performance-on-battlemage.md) (draft).
**Revisit when:** the machine changes (another card, more RAM), or a model family is added.

## Context

Strata runs on many PCs: NVIDIA or AMD, Windows or Linux, any of several model families. It keeps CUDA and HIP
builds in step, and has setup logic for each case. This engine is for one machine and one job.

## Decision

1. **One machine:**
   - GPUs: 2x Intel Arc Pro B60, 24 GB each (22.71 GiB usable), each on its own CPU root port at PCIe 5.0 x8.
   - CPU: 16 Zen 4 cores on 2 CCDs, with AVX-512.
   - RAM: 92 GB.
   - OS and driver: Linux, with the `xe` kernel driver.
   - No portability work for other GPUs, operating systems or CPUs.
2. **One model family:** Qwen3.8-Flash-Next, from ISTA-DASLab's GSQ-RCO GGUFs. Vision, Swift 1.5, the Coder and
   Unsloth's files are out of scope.
3. **The full context for every request:** 262,144 tokens.
4. **Several requests at once:** the target is 4 at full context, more at shorter ones.
5. **How it runs:** in Docker, as one model behind a model-swapping proxy such as llama-swap, with the two cards
   as an exclusive set, and with `--ulimit memlock=-1:-1` in its container.

## Consequences

- Kernels are compiled for one device ([ADR-0003](0003-kernel-language.md)). The specs cover one architecture
  (`qwen4exp`).
- Memory is budgeted for exactly two cards and this RAM ([MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md)).
- The engine shares the cards with the other models the proxy runs. It loads and unloads like them, and is measured
  with them unloaded ([ADR-0014](0014-how-work-is-done.md)).
