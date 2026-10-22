# ADR-0007: Q2_0 first, then the i-quants

**Status:** superseded by [ADR-0018](0018-one-file-iq3s.md) on 2026-10-10. Was accepted on 2026-10-09.
**Revisit when:** superseded.

## Context

Decode is bound by memory bandwidth, so the best format has few bytes per weight **and** a cheap unpack. On Xe2 the
unpack matters more than on NVIDIA. The i-quants decode through codebook lookups, which is where llama.cpp's SYCL
backend is slow on Battlemage ([llama.cpp #27517](https://github.com/ggml-org/llama.cpp/issues/27517)). That is a
kernel gap, not a hardware limit.

## Decision

1. **Q2_0 is the first model run end to end.**
2. **Then IQ2_XS and IQ3_XXS; IQ3_S last.**
3. **The expert kernels are designed for Q2_0 and IQ3_XXS together from the start,** so the i-quants' codebook
   lookups shape the design.
4. **Each expert is stored in a layout that gives wide, contiguous loads** (quants in one run, scales in another),
   whatever the file's layout. llama.cpp's "reorder" layout gained 21-87% on Intel for Q4_0.

## Why

- **Q2_0 is the fastest model Strata measured:** 93 tokens/s against 79 (IQ2_XS) and 62 (IQ3_XXS) on an RTX 5070;
  60 against 52 (IQ2_XS) on an RX 9070 XT.
- **Q2_0's unpack is the simplest:** plain 2-bit values plus a scale, then the integer dot instruction. There are no
  gathers.
- **Most of its ~34 GB of experts should fit on the two cards.**
- **Quality,** by Strata's labels: Q2_0 "good", IQ2_XS "a bit better", IQ3_XXS "great".

## Consequences

- The quant format spec ([QUANT_FORMATS.md](../spec/QUANT_FORMATS.md)) covers Q2_0 and IQ3_XXS first.
- The arena is repacked from the GGUF at load time, or once ahead of time.
