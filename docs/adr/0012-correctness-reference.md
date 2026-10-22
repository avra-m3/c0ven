# ADR-0012: correctness - llama.cpp's CPU backend is the reference

**Status:** accepted, 2026-10-09.
**Revisit when:** llama.cpp's implementation of this architecture is found to be wrong. The model's paper and the
specs in [docs/spec/](../spec/) then decide.

## Context

The engine starts from scratch, without Strata's parity tests ([ADR-0001](0001-level-zero-from-scratch.md)).
Strata's output was checked against llama.cpp. llama.cpp's SYCL backend is not a safe reference on Xe2: its reorder
path gave wrong output on a B70 ([llama.cpp #21893](https://github.com/ggml-org/llama.cpp/issues/21893)).

## Decision

1. **The reference is llama.cpp's CPU backend** at a pinned commit: Strata's `3cf03257f219afbe7334045ff7c6a06ac68c627d`.
   Never its SYCL or Vulkan backend.
2. **End to end:** the engine is correct when, on the same GGUF, it matches the reference in greedy-token agreement
   and in mean KL divergence. The gate is mean KL ~0.02, the paper's. The required agreement rate is set from the
   reference outputs ([REFERENCE_OUTPUTS.md](../measurements/REFERENCE_OUTPUTS.md)). Strata's one published figure
   (97.5-99% of positions on short greedy answers, 90-91% after a 16K prompt) is for Unsloth's UD-Q4_K_XL file, not
   these files.
3. **Per layer:** the reference's intermediate tensors for fixed inputs.
4. **Per kernel:** reference values from ggml's CPU implementation, or a scalar reference inside the test.
5. **A test whose reference data is missing skips with a message. It never passes.**

## Consequences

- **The reference data exists before the engine code it checks.** It is regenerated only when the pinned commit
  changes.
- **Kernel pull requests show their reference test passing on a B60,** since Level Zero has no CPU device.
