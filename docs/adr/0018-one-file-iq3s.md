# ADR-0018: one model file, GSQ-RCO IQ3_S

**Status:** accepted, 2026-10-10. Supersedes [ADR-0007](0007-quant-formats.md), and item 6 of
[ADR-0015](0015-ngram-table-on-the-ssd.md). Its second "revisit when" (RAM) was met on 2026-10-13 and answered by
[ADR-0023](0023-two-cards-tensor-split.md) item 5: the arena holds only the experts the cards do not cache, and the
file stays. Item 1 superseded by [ADR-0038](0038-the-other-quants-load.md) (draft).
**Revisit when:**

- the disk gains room for a second file;
- IQ3_S's arena and the streamed KV do not fit in the free RAM
  ([memory budget](../measurements/MEMORY_BUDGET.md),
  [host memory record](../measurements/MEASUREMENTS.md#2026-10-12-host-memory));
- the i-quant kernels stay far below the bandwidth the dense formats reach, so that a faster file is worth its
  quality.

## Context

- **The disk holds one file.**
- **ISTA-DASLab's repository has four sizes** (model card, checked 2026-10-10). Shard 2, the 28.8 GB n-gram table, is
  the same file in all four:

  | File | bpw | Shard 1 | Total | AIME25 | GPQA-D | LCB v6 | Task avg |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | BF16 (base) | 16.00 | - | 354 GB | 100.00 | 91.92 | 87.43 | 93.12 |
  | Q2_0 | 2.40 | 37.6 GB | 66.4 GB | 96.67 | 89.39 | 81.14 | 89.07 |
  | IQ2_XS | 2.50 | 39.2 GB | 68.0 GB | 96.67 | 87.37 | 83.43 | 89.16 |
  | IQ3_XXS | 3.00 | 47.0 GB | 75.8 GB | 100.00 | 91.41 | 86.29 | 92.57 |
  | **IQ3_S** | 3.50 | 54.8 GB | 83.6 GB | 100.00 | 92.93 | 86.86 | **93.26** |

- **A GSQ-RCO file is not one format.** RCO chooses a type per tensor; the name gives the size class. The repository's
  allocation file for IQ3_S (`tensor-allocation/...IQ3_S-00001-of-00002.rco-allocation.txt`) lists, for the routed
  experts:

  | Expert tensors | Types (layers) |
  | --- | --- |
  | gate and up | IQ2_S (20), IQ3_XXS (17), IQ3_S (10), IQ4_XS (1: layer 47) |
  | down | IQ4_NL (39), Q2_0 (9: layers 1, 8, 10, 12, 13, 20, 21, 30, 36) |

  The dense tensors are Q4_K, Q5_K, Q6_K, IQ4_XS, IQ4_NL and Q8_0, with BF16, F16 and F32 for the small ones.
  [tensors-IQ3_S.tsv](../model/tensors-IQ3_S.tsv) lists every tensor in the file.

## Decision

1. **The engine runs one file: ISTA-DASLab's GSQ-RCO IQ3_S,** at a pinned revision. It is the best of the four, or
   level with IQ3_XXS, on every benchmark reported for it (the zero-shot average is not reported for IQ3_S). No other
   quant is downloaded, measured or supported.
2. **The expert kernels cover every type the file's experts use:** IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS for gate and up,
   IQ4_NL and Q2_0 for down. A layer's gate and up share a type, so each layer needs one gate/up kernel variant and
   one down variant.
3. **Each expert is stored in a layout that gives wide, contiguous loads** (quants in one run, scales in another),
   whatever the file's layout. Carried over from ADR-0007: llama.cpp's "reorder" layout gained 21-87% on Intel for
   Q4_0.
4. **The n-gram table held in RAM is no longer a standing A/B arm** (ADR-0015 item 6). Beside IQ3_S's ~50 GB arena it
   fits only for one short request with no streamed KV (~87 of 92 GB, *estimate*). If it does not fit, the SSD path
   is measured against itself with the read-delay switch.

## Why

- **Quality comes first** for the one file the disk holds.
- **IQ3_S costs ~7.8 GB more than IQ3_XXS** for +1.52 on GPQA-Diamond and +0.57 on LiveCodeBench v6. The disk and
  the RAM (~71 of 92 GB, *estimate*) still have room for it.

## Options not chosen

| Option | Why not |
| --- | --- |
| Q2_0 (ADR-0007's first model) | the fastest file (Strata: 93 against 62 tokens/s for IQ3_XXS on an RTX 5070), but 4.2 points below IQ3_S on the task average |
| IQ3_XXS | 0.69 points below IQ3_S for 7.8 GB less; the disk does not need the saving |
| Two files | the disk holds one |

## Consequences

- **Fewer experts fit on the cards:** ~58% of IQ3_S's expert bytes, against ~85% for Q2_0 (*estimate*;
  [MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md)). The hit rate, the CPU's share of the experts and prefetching
  matter more.
- **The expert kernels are i-quant kernels from the start.** That is where llama.cpp's SYCL backend is slow on
  Battlemage ([llama.cpp #27517](https://github.com/ggml-org/llama.cpp/issues/27517)), so the i-quant gathers are on
  the critical path.
- **Strata's speed figures are for other files,** so the engine's speed estimates are made for IQ3_S.
- **The quant format spec** ([QUANT_FORMATS.md](../spec/QUANT_FORMATS.md)) covers the types in this file, not a list
  chosen in advance.
