# ADR-0029: the engine process turns on drafting and adaptive swaps

**Status:** accepted, 2026-10-16. Settles part of item 14 of
[ADR-0027](0027-the-c-abi-as-the-core-was-optimised.md). Item 3 superseded in part by [ADR-0050](0050-dflash-2-drafts.md) (draft).
**Revisit when:** a request needs tokens identical to another run's without `--deterministic`, or drafting or swaps
measure slower on a workload the engine serves.

## Context

- `engine_api.h` version 2 can draft with the model's MTP layer (`mtp_dir`) and can let the cached experts follow
  the routing (`ENGINE_CONFIG_SWAPS`). Without this ADR the adapter sets neither (ADR-0027, item 14).
- `OpenConfig.deterministic` is [ADR-0016](0016-how-a-change-is-judged.md)'s switches: no PCIe share of the misses,
  no adaptive swaps, no prefetching. Without this ADR it does not reach the core.
- At a 262,144-token context the cards cache 267-288 of each layer's 512 experts, and the CPU computes the rest:
  68 experts per token with the cached set fixed by the routing profile. Each row of a verify window adds about 70
  more, and a row costs about 5 ms against a plain step's 19
  ([the draft rounds' record](../measurements/MEASUREMENTS.md#2026-10-16-draft-rounds)). With swaps the CPU computed 10-16 experts
  per token after the first few hundred tokens, and a row cost about 4 ms.
- A draft never changes the tokens: the main model verifies every one, and the engine samples from the main model's
  candidates only. Swaps change which experts the CPU computes, so their rounding, not the arithmetic.
- The draft layer's tensors are not in the GGUF. [MODEL_FILES.md](../MODEL_FILES.md#the-mtp-layer) puts them in
  `mtp/` beside the model files, with `mtp-manifest.json`.

## Decision

1. The native backend asks for adaptive swaps unless `OpenConfig.deterministic` is set.
2. `OpenConfig` carries the draft layer's directory, `draft_dir`; empty means no drafting. The native backend passes it
   as `mtp_dir`.
3. The engine process gives `draft_dir` from `--draft-dir`, or else from the model module, which finds `mtp/` with
   its manifest beside the model file. `--no-draft` turns drafting off. Only the native backend drafts.
4. Drafting is not one of ADR-0016's switches: with `--deterministic`, the engine still drafts.

## Why

Through the C ABI at the model's own sampling, decode went from 54.76 to 65.31 tokens/s on long-4k and from about
54.7 to 66.99 on long-32k, the prompt losing 2.1% and 1.6% ([the record](../measurements/MEASUREMENTS.md#2026-10-16-drafting-and-swaps)).
Without swaps, drafting gave 5-13%; swaps alone help every step, and drafting gives the rest.

## Options not chosen

- Drafting or swaps alone. Without swaps, a window's extra rows cost the CPU about 70 experts each.
- Making drafting one of the deterministic switches. A draft does not change a token's arithmetic, only how many
  rows a step has, which ADR-0016's first rule already makes harmless.

## Consequences

- ADR-0027's item 14 is settled for swaps and drafting. The routing profile, the card count, the reserved memory and a
  fixed experts-per-layer count stay at the core's defaults, and remain open.
- The draft layer's memory leaves fewer experts cached: 303 per layer against 324 at a 262,144-token context. The
  prompt path streams the others in for every chunk, which cost long-4k's and long-32k's prompts 2.1% and 1.6%.
- Tokens are reproducible only with `--deterministic`: with swaps, which experts the CPU computes depends on what the
  engine served before.
