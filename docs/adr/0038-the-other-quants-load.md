# ADR-0038: the other three GSQ-RCO quants load

**Status:** draft, 2026-10-19; not accepted. Would supersede item 1 of [ADR-0018](0018-one-file-iq3s.md);
ADR-0018's other items stand.
**Revisit when:** ISTA-DASLab republishes the files with other types; a fifth size appears; a type's kernel falls
behind the others' speed for its bytes.

## Context

- [ADR-0018](0018-one-file-iq3s.md) item 1 chose IQ3_S for its quality, with the disk holding one file, and said no
  other quant is downloaded, measured or supported.
- The same repository has three other sizes: IQ3_XXS, IQ2_XS and Q2_0. Their second shard, the n-gram table, is byte
  for byte IQ3_S's, so a second size costs only its first shard (37.6-47.0 GB).
- RCO chooses types per tensor. Beside the six that ADR-0018 item 2 covers, the three files' experts use IQ2_XXS,
  IQ2_XS and IQ1_M (IQ2_XS's file only), and Q2_0's file stores dense tensors as Q3_K, Q4_0 and Q5_0. It also
  stores `blk.1.ple_key` as Q2_0, a tensor the engine reads in BF16.
- Measured against IQ3_S on two cards ([record](../measurements/MEASUREMENTS.md#2026-10-19-other-quants)): more experts fit on the
  cards (369, 460 and 481 of 512 a layer, against 307). Decode after a 4k prompt is 16-22% faster and prompt reading
  4-12% faster. Greedy agreement with IQ3_S's references is 89.9-93.6%, against IQ3_S's own 98.0%.

## Decision

1. **The engine loads any of ISTA-DASLab's four GSQ-RCO sizes** at the pinned revision. IQ3_S stays the default and
   the file every reference, gate and README figure is taken with.
2. **The expert kernels cover the types the four files' experts use:** ADR-0018 item 2's six, plus IQ2_XXS, IQ2_XS
   and IQ1_M for gate and up. Each is stored in the runs layout of ADR-0018 item 3. IQ1_M is decoded as whole numbers
   in -9..9 with two scales a unit. Q2_0 gate and up, which only Q2_0's file has, run on the dense matrix kernels,
   because the expert kernel's gate/up path assumes 256 values a block.
3. **The dense kernels add Q3_K, Q4_0 and Q5_0,** which Q2_0's file uses for dense tensors.
4. **A quantised tensor the engine reads in BF16 is widened when loaded.**
5. **The kernel tests check every quantised type of the file they open** (`FLASH_MODEL`). `gguf_test` and
   `greedy_test`'s gate stay on IQ3_S's numbers.

## Why

- The files differ only in their types. The engine already chose types per tensor, so the cost is the types'
  decoding, not a second design.
- The smaller files trade a few points of agreement for speed. Each deployment chooses; the default stays the best
  file.

## Options not chosen

| Option | Why not |
| --- | --- |
| Keep IQ3_S only (ADR-0018 item 1) | the smaller files are 16-22% faster at decode, for a few points of agreement |
| Every type llama.cpp has | the types no file of this model uses would be untested code |

## Consequences

- Kernel builds and the type tables grow by six types. `docs/spec/QUANT_FORMATS.md` covers them.
- MEASUREMENTS.md's figures and the README's stay on IQ3_S. The other files' figures are in their own table.
- `greedy_test` against the IQ3_S references fails its 95% gate on another file, as it should: its references are
  IQ3_S's.
