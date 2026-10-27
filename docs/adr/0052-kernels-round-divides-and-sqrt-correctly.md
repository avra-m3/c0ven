# ADR-0052: kernels round fp32 divides and square roots correctly

**Status:** draft, 2026-10-28; not accepted. Supersedes item 2 of [ADR-0003](0003-kernel-language.md) in part (the
kernels' build options).
**Revisit when:** a kernel's time is bound by divides or square roots, or an IGC update rounds them correctly without
the option.

## Context

- OpenCL C lets a single-precision divide be off by up to 2.5 ulp and a square root by up to 3 ulp unless the
  program is built with `-cl-fp32-correctly-rounded-divide-sqrt`. IGC uses that freedom on the B60.
- The kernels' quantizers compute ggml's Q8_0 and q8_1 rule: the scale `d = max|x| / 127`, then `id = 1 / d` and the
  codes `round(x * id)` (`kernels/include/a8.h`, `quant_q8_0` in `layer.cl` and `prompt.cl`). The norms divide a sum
  of squares by the row length. ggml computes the same on the CPU with correctly rounded IEEE divides and square
  roots, and llama.cpp's CPU backend is the reference ([ADR-0012](0012-correctness-reference.md)).
- Built without the option, 4.7% of `x / 127` results, 8.2% of `sqrt(x)` and 4.3% of Q8_0 blocks (their scale or
  codes) differ from the correctly rounded result on a B60. Built with it, none do
  ([the record](../measurements/MEASUREMENTS.md#2026-10-27-divide-sqrt-rounding)).
- Strata's SYCL build passes the same option, for the same quantizers to match ggml byte for byte
  ([`b88eff6`](https://github.com/avra-m3/strata-intel/commit/b88eff688b531a4edea718af1bca331afbac5fe4), `sycl/CMakeLists.txt`).

## Decision

1. Every kernel module is built with `-cl-fp32-correctly-rounded-divide-sqrt`, beside `-cl-std=CL3.0`
   (`cmake/kernels.cmake`).
2. A kernel that wants a faster, approximate divide or square root calls `native_divide` or `native_sqrt` itself,
   with a record that shows the gain and `greedy_test`'s agreement.

## Why

The kernels then quantize activations and the KV cache to ggml's bytes, so a difference from the reference comes
from the order of sums alone. `greedy_test` agrees with llama.cpp on 847 of 861 positions against 843, and on 844
against 843 with prompts read in chunks of 512. Decode of doc-4k-qa costs 15.62 → 15.69 ms a token and prompt
reading 2,967 → 2,957 tokens/s ([the record](../measurements/MEASUREMENTS.md#2026-10-27-divide-sqrt-rounding)).

## Options not chosen

- **The option on the quantizers' modules only.** The norms in the same modules divide too, and the cost over all
  modules is 0.5% of a decode step.
- **Multiplying by a reciprocal computed once.** It gives other bytes than ggml's `amax / 127`.

## Consequences

- Every kernel module is rebuilt. The option is in each module's build string, so each binary differs from one built
  before this decision.
- A change to a kernel's divides or square roots is judged by `greedy_test` like any change to its sums
  ([ADR-0016](0016-how-a-change-is-judged.md)).
