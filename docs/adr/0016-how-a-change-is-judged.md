# ADR-0016: a change is judged by the whole window; a token's arithmetic does not depend on the batch

**Status:** accepted, 2026-10-09. The rules come from [Splash](https://github.com/incoai/splash) and Strata's own
rule in `include/strata/core/verify.hpp`.
**Revisit when:** no trigger is expected. A measurement could make one switch below unnecessary, but not the rule.

## Context

- **Speculative decoding makes rounding visible.** A draft token is accepted when it matches what greedy decode
  would have chosen. A change that moves rounding moves acceptance, and with it the tokens per window, even when it
  passes every numerical test.
- **Splash measured this.** It rejected two changes that had passed its numerical tests, because they lowered
  acceptance on some prompts, and with it throughput: a parallel reduction of the attention's softmax denominator, and
  split tiles for one-request projections (`dev/benchmarks/remaining-decode-optimizations.md`,
  `dev/benchmarks/device-policy.md` in Splash).
- **GPU timings miss the host.** The host's work between commands is part of every window. Splash judges speed by the
  wall time of a whole decode step (`metrics.decode_cycle_ms`).
- **Strata's rule:** every multi-token kernel's arithmetic for a token is the single-token kernel's, bit for bit
  (`verify.hpp`). Splash's rule for splitting a projection's sums depends on the grid only, never on the batch
  width.
- **The engine has a source of difference neither rule covers.** Which missed experts go over PCIe and which to the
  CPU depends on the misses of the whole batch, and the two round an expert differently. Prefetching and adaptive
  swaps move experts between them the same way. Strata needs `--pcie-frac 0 --adapt-swaps 0` for identical tokens,
  and `--prompt-cache 0` for byte-identical repeats (its `docs/DETAILS.md`, #410).

## Decision

1. **A token's arithmetic does not depend on what else is in the window or the batch.** How a kernel splits a sum
   depends on its grid, never on the number of tokens or sequences.
2. **The engine has switches that make a run reproducible:** no PCIe share of the misses, no adaptive swaps, no
   prefetching, no prompt cache. With them, a request's tokens are identical alone, in a window, batched and pipelined.
   Every check that compares tokens sets them.
3. **Every change to a kernel or to the decode loop records two numbers:** accepted tokens per window, and the wall
   time per window.
4. **Before and after runs alternate (ABBA) on fresh processes.** Medians are reported, with their range.
5. **A change that is faster alone but lowers acceptance is rejected.**

## Consequences

- Kernel designs that split sums by the number of tokens (split-K chosen per batch width, for one) are ruled out.
- The tests have a determinism level: alone against batched, with the switches set.
- The engine reports accepted tokens per window and wall time per window.
- [How the numbers were taken](../measurements/MEASUREMENTS.md#how-the-numbers-were-taken) describes an ABBA run.
