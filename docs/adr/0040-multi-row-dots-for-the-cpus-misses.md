# ADR-0040: the CPU decodes a missed expert's weights once for all the rows that need it

**Status:** draft, 2026-10-19; not accepted. Would depart from item 2 of [ADR-0004](0004-write-or-reuse.md) for four
types.
**Revisit when:** the model gains an expert type that a step's misses spend much time on and that has no multi-row
dot here, or the reference moves to a ggml whose AVX2 dot products differ from the copied steps.

## Context

- ADR-0004 takes the CPU's expert products from ggml, and names written kernels for several tokens at once as a
  later optimisation. ggml dots one activation row at a time: a missed expert that a step's 2-4 rows share has each
  of its weight blocks decoded once per row
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-11-cpu-bar)). A further row costs 19-25 µs against 60-62 µs for the
  expert's first ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-16-cpu-answer-per-layer)).
- Drafting makes most decode steps 2-4 rows, and card 0 waits for the CPU longest at four
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-16-cpu-answer-per-layer)).
- The model's expert types are IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS for gate and up, IQ4_NL and Q2_0 for down.

## Decision

1. For IQ2_S, IQ3_XXS, IQ3_S and IQ4_NL, `compute_many` dots each weight row of an expert that several jobs share
   against all their activation rows in one call (`core/cpu/multirow.cpp`), up to four rows per pass over the row.
2. Each row's sum takes ggml's AVX2 steps for the type in the same order, so its bits equal ggml's `vec_dot`.
   `multirow_test` checks them for 1 to 9 rows.
3. The functions need AVX2, FMA and F16C, checked at run time on the CPU. Without them, and for every other type,
   the CPU dots each row with ggml as before.

## Why

On one core, against ggml's loop over rows, one 640 × 2,560 matrix of IQ2_S, IQ3_XXS or IQ3_S takes 1.72-1.78
times less time at 2 rows, 2.22-2.35 at 3 and 2.35-2.66 at 4; IQ4_NL's 2,560 × 640 down 1.15, 1.27 and 1.39. Decode
over eleven prompts was 2.4-2.6% faster, with the same tokens
([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-19-multi-row-misses)).

## Options not chosen

- AVX-512 kernels: ggml's AVX2 steps give the reference's bits as they are; other steps would change the rounding of
  every miss.
- IQ4_XS and Q2_0: IQ4_XS is one layer's gate and up; ggml has no SIMD path for Q2_0 to copy bit for bit.

## Consequences

The steps copied from ggml are tied to the pinned commit of [ADR-0012](0012-correctness-reference.md):
`multirow_test` fails if a new ggml changes them.
