# Quant formats: every weight and cache format the engine reads

> The engine's kernels decode the weights' bits themselves. This page says what each format's bits mean: the block,
> the formula back to a real number, the integer dot product llama.cpp uses, and how a block is laid out for wide
> loads. It covers every type in the census of the IQ3_S file ([MODEL_FILES.md](../MODEL_FILES.md)), Q8_0 for the
> 8-bit KV cache ([ADR-0008](../adr/0008-kv-cache.md)) and the activation formats. No llama.cpp or Strata code is
> copied: the page uses prose and tables, citing file and line at a pinned commit.

**Source.** llama.cpp at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`3cf0325`, 2026-09-20, "CUDA: enable
sparse fa for qwen4 (#28770)"). It is Strata's pin (`setup.py:83`, `third_party/ggml/VERSION.txt` at `1678de3`), the
commit [MODEL_LAYER.md](MODEL_LAYER.md) uses and the build the reference outputs were made with. Paths below are in that tree:

| Short name | Path |
| --- | --- |
| `common.h` | `ggml/src/ggml-common.h` (git blob `1dbbe326`, sha256 `0061131b…595c`) |
| `quants.c` | `ggml/src/ggml-quants.c` |
| `vecdotq.cuh` | `ggml/src/ggml-cuda/vecdotq.cuh` |
| `mmvq.cu`, `quantize.cu`, `fattn-common.cuh`, `cpy-utils.cuh` | `ggml/src/ggml-cuda/` |
| `vecdotq.hpp`, `ggml-sycl.cpp` | `ggml/src/ggml-sycl/` |
| `ggml-cpu.c` | `ggml/src/ggml-cpu/ggml-cpu.c` |


**Conventions.**

- All multi-byte fields are **little-endian**. The file is a little-endian GGUF: `gguf-py` reads its version as 3 on a
  little-endian host.
- `f16` is IEEE binary16 (`ggml_half` is a `uint16_t` in C, `common.h:6`). `bf16` is the top 16 bits of an IEEE
  binary32: widen by shifting left 16 (`ggml/src/ggml-impl.h:606-613`).
- A *block* is the unit the file stores; a tensor row is a whole number of blocks. A *super-block* is a block of
  256 values (`QK_K`, `common.h:89`) split into *sub-blocks* of 16 or 32. Values are numbered 0..n-1 within a block.
- "Field at offset o, n B" is a byte offset from the block's first byte. Bits are numbered from the least
  significant.
- bpw = bytes per block x 8 / values per block.

## In short

| Type | ggml id | Where in the file | Values / block | Bytes / block | bpw | Codebook | GPU dot product (llama.cpp CUDA) | CPU activation type |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| [IQ2_S](#iq2_s) | 22 | experts gate+up (20 layers) | 256 | 82 | 2.5625 | `iq2s_grid`, 1024 x 8 B | q8_1, DP4A | q8_K |
| [IQ3_XXS](#iq3_xxs) | 18 | experts gate+up (17 layers) | 256 | 98 | 3.0625 | `iq3xxs_grid`, 256 x 4 B; `ksigns_iq2xs` | q8_1, DP4A | q8_K |
| [IQ3_S](#iq3_s) | 21 | experts gate+up (10 layers) | 256 | 110 | 3.4375 | `iq3s_grid`, 512 x 4 B | q8_1, DP4A | q8_K |
| [IQ4_XS](#iq4_xs) | 23 | experts gate+up (layer 47), 42 dense, embedding | 256 | 136 | 4.25 | `kvalues_iq4nl`, 16 x 1 B | q8_1, DP4A | q8_K |
| [IQ4_NL](#iq4_nl) | 20 | experts down (39 layers), 47 dense, n-gram table | 32 | 18 | 4.5 | `kvalues_iq4nl`, 16 x 1 B | q8_1, DP4A | q8_0 |
| [Q2_0](#q2_0) | 42 | experts down (9 layers) | 64 | 18 | 2.25 | none | q8_1, DP4A | q8_0 |
| [IQ2_XXS](#iq2_xxs) | 16 | not in the IQ3_S file; experts gate+up in the IQ3_XXS and IQ2_XS files | 256 | 66 | 2.0625 | `iq2xxs_grid`, 256 x 8 B | - | q8_K |
| [IQ2_XS](#iq2_xs) | 17 | not in the IQ3_S file; experts gate+up in the IQ3_XXS file | 256 | 74 | 2.3125 | `iq2xs_grid`, 512 x 8 B | - | q8_K |
| [IQ1_M](#iq1_m) | 29 | not in the IQ3_S file; experts gate+up in the IQ2_XS file | 256 | 56 | 1.75 | `iq1s_grid`, 2048 x 8 B | - | q8_K |
| [Q3_K](#q3_k) | 11 | not in the IQ3_S file; dense and the embedding in the Q2_0 file | 256 | 110 | 3.4375 | none | - | q8_K |
| [Q4_K](#q4_k) | 12 | 47 dense | 256 | 144 | 4.5 | none | q8_1, DP4A | q8_K |
| [Q5_K](#q5_k) | 13 | 35 dense | 256 | 176 | 5.5 | none | q8_1, DP4A | q8_K |
| [Q6_K](#q6_k) | 14 | 128 dense, the head | 256 | 210 | 6.5625 | none | q8_1, DP4A | q8_K |
| [Q8_0](#q8_0) | 8 | 1 dense (`blk.47.ffn_down_shexp`); the KV cache; in Unsloth's UD-IQ4_XS file, experts down (5 layers) and most dense tensors | 32 | 34 | 8.5 | none | q8_1, DP4A | q8_0 |
| [Q4_0](#q4_0-and-q5_0) | 2 | not in the IQ3_S file; 16 dense in the Q2_0 file | 32 | 18 | 4.5 | none | - | q8_0 |
| [Q5_0](#q4_0-and-q5_0) | 6 | not in the IQ3_S file; 7 dense in the Q2_0 file | 32 | 22 | 5.5 | none | - | q8_0 |
| [BF16](#bf16-f16-f32) | 30 | 484 (dense, head, PLE) | 1 | 2 | 16 | none | float kernels | bf16 |
| [F16](#bf16-f16-f32) | 1 | 1 (`blk.1.ple_conv1d`) | 1 | 2 | 16 | none | float kernels | f16 |
| [F32](#bf16-f16-f32) | 0 | 292 (norms and other small tensors) | 1 | 4 | 32 | none | float kernels | f32 |
| [q8_1](#q8_1-the-gpu-activation-format) | 9 | activations only (GPU) | 32 | 36 | 9 | none | the other operand | - |
| [q8_K](#q8_k-the-cpu-activation-format-for-k--and-i-quants) | 15 | activations only (CPU) | 256 | 292 | 9.125 | none | - | the other operand |

The tensor counts are from the census in [MODEL_FILES.md](../MODEL_FILES.md). Each of the file's 1,224 tensors has
bytes = values / values-per-block x bytes-per-block with this table, and every row is a whole number of blocks
(checked against [tensors-IQ3_S.tsv](../model/tensors-IQ3_S.tsv); see [Sources](#sources-and-how-this-was-checked)).

The rows marked "not in the IQ3_S file" are the types the other quants of the same repository add (the IQ3_XXS,
IQ2_XS and Q2_0 folders, by their `tensor-allocation/*.rco-allocation.txt` lists); the engine reads them too, and their
sections describe its own decoding rather than llama.cpp's GPU dot product.

The CPU column is each type's `vec_dot_type` in `ggml-cpu.c` (Q2_0 `:234`, Q4_0 `:243`, Q5_0 `:263`, Q8_0 `:272`,
Q3_K `:308`, Q4_K `:311`, Q5_K `:321`, Q6_K `:327`, IQ2_XXS `:340`, IQ2_XS `:346`, IQ3_XXS `:349`, IQ3_S `:356`,
IQ2_S `:362`, IQ1_M `:377`, IQ4_NL `:380`, IQ4_XS `:386`, BF16 `:395`, F16 `:222`). It
matters because the correctness reference is llama.cpp's CPU backend ([ADR-0012](../adr/0012-correctness-reference.md),
[REFERENCE_OUTPUTS.md](../measurements/REFERENCE_OUTPUTS.md)). There the activation meets a K- or i-quant as q8_K, not q8_1.

## How the GPU dot products work, in general

llama.cpp's matrix-vector kernel (MMVQ, `mmvq.cu`) never turns weights into floats. Its steps:

1. **Quantizes the activation vector to q8_1** once per matrix-vector product: 32 values per block, an f16 scale and an
   f16 sum ([q8_1](#q8_1-the-gpu-activation-format)). The vector is padded to a multiple of 512 values
   (`MATRIX_ROW_PADDING`, `common.cuh:186`; used at `mmvq.cu:1484`).
2. **Walks a weight row block by block.** Each thread takes `vdr` 32-bit words of one block, starting at word
   `iqs` (`mmvq.cu:699-703`; `vdr` per type at `mmvq.cu:69-94`, from the `VDR_*_MMVQ` defines in `vecdotq.cuh`).
3. **Builds 4 signed 8-bit weights per 32-bit word** (unpacking bits, looking up a codebook, applying signs) and
   multiplies them with 4 activation bytes using **DP4A**: a 4-way int8 dot product added to an int32
   (`ggml_cuda_dp4a`, `common.cuh:720`).
4. **Scales once per q8_1 block:** the int32 sum times the weight's (sub-)block scale times the activation block's
   scale `d`. The float partial sums are reduced across the threads of a row.

The general shape, for one weight block or sub-block meeting one q8_1 block of 32 activations a_i = d_a q_i:

  Σ w_i a_i = (d_w s) · d_a · Σ c_i q_i  (one DP4A chain for Σ c_i q_i)

where c_i is the weight's small integer (code, or codebook value with its sign) and s its sub-block scale. Formats
with a per-block *minimum* (Q4_K, Q5_K) add a second term, d_min m · d_a · Σ q_i, from the activation's sum.

The SYCL backend has the same functions (`vecdotq.hpp`: Q2_0 `:1006`, Q8_0 `:1155`, Q4_K `:1225`, Q5_K `:1285`, Q6_K
`:1375`, IQ2_S `:1492`, IQ3_XXS `:1546`, IQ3_S `:1588`, IQ4_NL `:1694`, IQ4_XS `:1721`). Where it differs from CUDA in
arithmetic, the type's section says so.

Threads per block below means MMVQ threads that share one weight block (`qi / vdr`, with `qi` the block's 32-bit
words of quants as `common.cuh` defines them).

## The expert types

### IQ2_S

256 values in 82 B, 2.5625 bpw (`block_iq2_s`, `common.h:395-402`). Eight sub-blocks of 32; each sub-block is four
groups of 8 values. A group is one entry of a 1024-entry codebook, plus 8 sign bits.

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 32 | `qs[0..31]` | low 8 bits of each group's 10-bit grid index; byte 4b+g is group g of sub-block b |
| 34 | 32 | `qs[32..63]` (signs) | one byte per group, same order; bit j set means value j of the group is negative |
| 66 | 8 | `qh[8]` | one byte per sub-block; bits 2g and 2g+1 are bits 8-9 of group g's index |
| 74 | 8 | `scales[8]` | one byte per sub-block: low nibble scales values 0-15, high nibble values 16-31 |
| | **82** | | 2 + 32 + 32 + 8 + 8 |

The struct names the 64-byte field `qs`; the second half holds signs (`quants.c:2554`).

**Dequantisation** (`dequantize_row_iq2_s`, `quants.c:2543-2571`). For value j (0-7) of group g in sub-block b:

- index = `qs[4b+g]` | (((`qh[b]` >> 2g) & 3) << 8) (`quants.c:2561`)
- s = the scale nibble for the half the group is in (groups 0-1: low, 2-3: high)
- **y = d · (0.5 + s) / 4 · grid[index][j] · (−1 if sign bit j of byte `qs[32+4b+g]` else +1)** (`quants.c:2557-2563`)

grid[index][j] is byte j of `iq2s_grid[index]`, a `uint64_t`; every byte is 8, 25 or 43.

**Dot product** (`vec_dot_iq2_s_q8_1`, `vecdotq.cuh:1124-1167`; `vdr` 2, 8 threads per block, one sub-block per
thread). For each group: read the 8 grid bytes as two words, make each byte negative where its sign bit is set (XOR
with a 0xFF-per-byte mask, then subtract the mask), DP4A with the activations. Values 0-15 go into sum0 and 16-31 into
sum1. Then, in integers, **sumi = (sum0·s0 + sum1·s1 + (sum0 + sum1)/2) / 4** (`vecdotq.cuh:1163`), and the result is
d · d_a · sumi. That is (s + 0.5)/4 with integer division, so it truncates. SYCL instead computes
d · d_a / 4 · ((0.5 + s0)·sum0 + (0.5 + s1)·sum1) in floats (`vecdotq.hpp:1535-1536`): the two backends round
differently.

### IQ3_XXS

256 values in 98 B, 3.0625 bpw (`block_iq3_xxs`, `common.h:404-411`). Eight sub-blocks of 32; each is eight groups of
4 values, one 256-entry codebook entry each. Signs come in fours of 7 bits, the 8th sign implied by parity.

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 64 | `qs[0..63]` | 8-bit grid index per group of 4; byte 8b+k is group k of sub-block b |
| 66 | 32 | `qs[64..95]` | eight little-endian `uint32`, one per sub-block: bits 7l..7l+6 (l = 0-3) are a sign index for values 8l..8l+7; bits 28-31 are the scale s |
| | **98** | | 2 + 64 + 32 |

**Dequantisation** (`quants.c:2575-2603`). For sub-block b with word w = the b-th `uint32`:

- s = w >> 28; **db = d · (0.5 + s) / 2** (`quants.c:2589`)
- for l = 0-3: signs = `ksigns_iq2xs[(w >> 7l) & 127]` (`quants.c:2591`), 8 sign bits for values 8l..8l+7
- values 8l..8l+3 come from `iq3xxs_grid[qs[8b+2l]]`, values 8l+4..8l+7 from `iq3xxs_grid[qs[8b+2l+1]]`
  (`quants.c:2592`)
- **y = db · grid byte · (−1 if the value's sign bit is set else +1)**

Each grid entry is a `uint32` of 4 bytes, each one of 4, 12, 20, 28, 36, 44, 52, 62. `ksigns_iq2xs[k]` is k with bit
7 set when k has an odd number of set bits, so every sign byte has even parity.

**Dot product** (`vecdotq.cuh:1172-1205`; `vdr` 2, 8 threads per block, one sub-block per thread). As IQ2_S, with
two 4-byte grid entries per 8 values. The sign byte is computed rather than looked up: the 8th bit is the 7-bit
index's parity (`unpack_ksigns`, `vecdotq.cuh:97-104`). Then **sumi = (s·sumi + sumi/2) / 2** in integers
(`vecdotq.cuh:1202`), result d · d_a · sumi. SYCL looks the signs up in `ksigns64` and scales in floats,
d · (0.5 + s) · d_a / 2 (`vecdotq.hpp:1557-1573`).

### IQ3_S

256 values in 110 B, 3.4375 bpw (`block_iq3_s`, `common.h:413-422`). Eight sub-blocks of 32, eight groups of 4
values each, one 512-entry codebook entry per group; signs stored plainly; one 4-bit scale per sub-block.

```
offset  0     2                         66      74                  106   110
        +-----+-------------------------+-------+-------------------+-----+
        |  d  | qs: 64 x index bits 0-7 |qh: 8  | signs: 32         |sc: 4|
        +-----+-------------------------+-------+-------------------+-----+
               byte 8b+k = group k        byte b  bytes 4b..4b+3      byte c: low nibble
               of sub-block b             bit k = values of sub-b.    sub-block 2c, high
                                          index   b, 1 bit each       nibble 2c+1
                                          bit 8
```

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 64 | `qs[64]` | low 8 bits of the 9-bit grid index per group of 4; byte 8b+k is group k of sub-block b |
| 66 | 8 | `qh[8]` | one byte per sub-block; bit k is bit 8 of group k's index |
| 74 | 32 | `signs[32]` | four bytes per sub-block; bit j of byte 4b+l is the sign of value 8l+j |
| 106 | 4 | `scales[4]` | byte c: low nibble is sub-block 2c's scale, high nibble sub-block 2c+1's |
| | **110** | | 2 + 64 + 8 + 32 + 4 |

**Dequantisation** (`quants.c:2607-2646`). For value i of group k in sub-block b:

- index = `qs[8b+k]` | (((`qh[b]` >> k) & 1) << 8) (`quants.c:2622`)
- s = scale nibble of sub-block b; **db = d · (1 + 2s)** (`quants.c:2619-2620`)
- **y = db · byte i of `iq3s_grid[index]` · (−1 if the value's sign bit is set else +1)** (`quants.c:2625`)

Each grid byte is an odd number from 1 to 15.

**Dot product** (`vecdotq.cuh:1211-1248`; `vdr` 2, 8 threads per block, one sub-block per thread). As IQ3_XXS, with
the signs read from `signs`. The scale is an exact integer: **sumi · (1 + 2s)** (`vecdotq.cuh:1244`), result
d · d_a · sumi.

### IQ4_XS

256 values in 136 B, 4.25 bpw (`block_iq4_xs`, `common.h:454-460`). Eight sub-blocks of 32 values; each value is a
4-bit index into a fixed 16-entry non-linear table; each sub-block has a 6-bit scale.

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 2 | `scales_h` (u16) | bits 2b, 2b+1: bits 4-5 of sub-block b's scale |
| 4 | 4 | `scales_l[4]` | byte b/2: low nibble is sub-block b's bits 0-3 when b is even, high nibble when b is odd |
| 8 | 128 | `qs[128]` | 16 bytes per sub-block; byte 16b+j: low nibble is value j, high nibble value j+16 |
| | **136** | | 2 + 2 + 4 + 128 |

**Dequantisation** (`quants.c:2743-2764`): ls = the 6-bit scale (`quants.c:2754`), **y = d · (ls − 32) ·
`kvalues_iq4nl`[nibble]** (`quants.c:2755-2758`). `kvalues_iq4nl` is 16 signed bytes from −127 to 113.

**Dot product** (`vecdotq.cuh:1357-1380`; `vdr` 4, 8 threads per block, one sub-block per thread). Each 32-bit word of
`qs` holds 8 nibbles; a byte-permute against the 16-byte table turns them into two words of 4 int8 values, one with
the low nibbles and one with the high (`get_int_from_table_16`, `vecdotq.cuh:34-95`). DP4A against activations 0-15
and 16-31; **sumi · (ls − 32)** (`vecdotq.cuh:1376`); result d · d_a · sumi.

### IQ4_NL

32 values in 18 B, 4.5 bpw (`block_iq4_nl`, `common.h:446-452`). The same table as IQ4_XS, one f16 scale per 32.

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 16 | `qs[16]` | byte j: low nibble is value j, high nibble value j+16 |
| | **18** | | 2 + 16 |

**Dequantisation** (`quants.c:2725-2741`): **y = d · `kvalues_iq4nl`[nibble]** (`quants.c:2735-2736`).

**Dot product** (`vecdotq.cuh:1333-1352`; `vdr` 2, 2 threads per block, 16 values each): table lookup as IQ4_XS, four
DP4A, result d · d_a · sumi.

The n-gram table is IQ4_NL too: a row of 160 values is 5 blocks, 90 B ([tensors-IQ3_S.tsv](../model/tensors-IQ3_S.tsv)).

### Q2_0

64 values in 18 B, 2.25 bpw (`block_q2_0`, `common.h:187-192`; `GGML_TYPE_Q2_0 = 42`, `ggml/include/ggml.h:432`). One
scale for 64 values, four 2-bit codes per byte. **Q2_0 spans two q8_1 blocks.**

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 16 | `qs[16]` | value j is bits 2(j mod 4) and 2(j mod 4)+1 of byte j/4 |
| | **18** | | 2 + 16 |

**Dequantisation** (`quants.c:439-457`): code q in 0..3, **y = (q − 1) · d** (`quants.c:454`). The codes mean
−1, 0, +1, +2: not symmetric.

**Dot product** (`vecdotq.cuh:728-772`; `vdr` 1, 2 threads per block, one 32-value half per thread, which meets one
q8_1 block). Each 16-bit piece of `qs` (8 codes) is spread to 8 bytes with a byte-permute whose table is the four bytes
−1, 0, 1, 2 (`vecdotq.cuh:758`), so the −1 offset is applied in the unpack and no activation sum is needed. Eight
DP4A, result **d · d_a · sumi** (`vecdotq.cuh:771`). On the CPU the activation is q8_0. Strata's CPU kernel uses
the identity Σ(q − 1)a = Σ q·a − Σ a with a per-chunk activation sum instead (`include/strata/kernels/cpu/expert.hpp:61`
at `1678de3`).

### IQ2_XXS

256 values in 66 B, 2.0625 bpw (`block_iq2_xxs`, `common.h:381-384`; `GGML_TYPE_IQ2_XXS = 16`, `ggml.h:406`).

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 64 | `qs[32]` (u16) | per 32 values, two 32-bit words: four 8-bit grid indices, then four 7-bit sign groups and a 4-bit scale |
| | **66** | | 2 + 64 |

**Dequantisation** (`quants.c:2488-2512`): for 32-value group g, with words a0 (indices) and a1, each 8 values l take
`iq2xxs_grid[byte l of a0]` (8 magnitudes of 8, 25 or 43), signs `ksigns_iq2xs[(a1 >> 7l) & 127]` (the 7 bits and an
eighth for even parity), and **y = d · (0.5 + (a1 >> 28)) · 0.25 · grid byte · sign**. The engine computes the parity
bit instead of reading the table, as for IQ3_XXS, and decodes a unit to 8 signed byte words with one scale.

### IQ2_XS

256 values in 74 B, 2.3125 bpw (`block_iq2_xs`, `common.h:388-392`; `GGML_TYPE_IQ2_XS = 17`, `ggml.h:407`).

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 64 | `qs[32]` (u16) | per 8 values: a 9-bit index into `iq2xs_grid` (bits 0-8) and 7 sign bits (bits 9-15) |
| 66 | 8 | `scales[8]` | per 32 values: a 4-bit scale for values 0-15 (low nibble) and one for 16-31 (high) |
| | **74** | | 2 + 64 + 8 |

**Dequantisation** (`quants.c:2516-2539`): **y = d · (0.5 + scale) · 0.25 · grid byte · sign**, the sign byte from
the 7 bits and the even-parity eighth (`ksigns_iq2xs`). The two halves of a unit have their own scale, which the
engine's unit takes as its two scales.

### IQ1_M

256 values in 56 B, 1.75 bpw (`block_iq1_m`, `common.h:433-437`; `GGML_TYPE_IQ1_M = 29`, `ggml.h:419`). **No field
holds the block scale:** it is the top nibbles of the four 16-bit scale words (`iq1m_scale_t`, `common.h:444`).

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 32 | `qs[32]` | per 8 values: the low 8 bits of an 11-bit index into `iq1s_grid` |
| 32 | 16 | `qh[16]` | per 8 values, a nibble: the index's top 3 bits and the sign of its delta (bit 3) |
| 48 | 8 | `scales[8]` | 4 words: per 32 values two 3-bit scales (bits 6(s mod 2) of word s/2); bits 12-15 of the words hold d |
| | **56** | | 32 + 16 + 8 |

**Dequantisation** (`quants.c:2675-2723`): d is the f16 whose nibbles are bits 12-15 of words 0, 1, 2, 3, from low to
high. For 8 values with grid entry g (each byte −1, 0 or 1) and delta δ = ±1/8 (`IQ1M_DELTA`, `common.h:1133`),
**y = d · (2 · scale + 1) · (g + δ)**, values 0-15 of a group with its first scale and 16-31 with its second. The engine
writes it as **(d · (2 · scale + 1) / 8) · (8g ± 1)**: whole numbers in −9..9, so a unit is 8 signed byte words and
two scales with no offset term.

## The dense types

### Q3_K

256 values in 110 B, 3.4375 bpw (`block_q3_K`, `common.h:315-320`; `GGML_TYPE_Q3_K = 11`, `ggml.h:401`).

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 32 | `hmask[32]` | the high bit of every value: value v's is bit v / 32 of byte v mod 32 |
| 32 | 64 | `qs[64]` | the low 2 bits: value v's are bits 2((v mod 128) / 32) of byte 32(v / 128) + v mod 32 |
| 96 | 12 | `scales[12]` | 16 6-bit scales: low nibbles in bytes 0-7 (scales 0-7 low, 8-15 high), top 2 bits in bytes 8-11 |
| 108 | 2 | `d` (f16) | block scale |
| | **110** | | 32 + 64 + 12 + 2 |

**Dequantisation** (`quants.c:1305-1353`): **y = d · (scale − 32) · (q − 4 · (1 − h))**, q the 2-bit code and h the
high bit; scale i covers values 16i..16i+15. Scale i's low nibble is byte i's (i < 8) or byte i − 8's high nibble,
and its top bits are bits 2(i / 4) of byte 8 + i mod 4 (the shuffle at `quants.c:1323-1328`). A 32-value unit s takes
scales 2s and 2s + 1.

### Q4_K

256 values in 144 B, 4.5 bpw (`block_q4_K`, `common.h:323-338`). Eight sub-blocks of 32, each with a 6-bit scale and
a 6-bit minimum: y = d·sc·q − dmin·m.

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | scale of the scales |
| 2 | 2 | `dmin` (f16) | scale of the minimums |
| 4 | 12 | `scales[12]` | eight 6-bit scales and eight 6-bit minimums, packed (below) |
| 16 | 128 | `qs[128]` | four chunks of 32 bytes; in chunk c, byte l's low nibble is value 64c+l (sub-block 2c), high nibble value 64c+32+l (sub-block 2c+1) |
| | **144** | | 2 + 2 + 12 + 128 |

The 12 scale bytes (`get_scale_min_k4`, `quants.c:880-887`):

| Bytes | Bits 0-5 | Bits 6-7 |
| --- | --- | --- |
| 0-3 | scale of sub-blocks 0-3 | bits 4-5 of the scale of sub-blocks 4-7 |
| 4-7 | minimum of sub-blocks 0-3 | bits 4-5 of the minimum of sub-blocks 4-7 |
| 8-11 | low nibble: bits 0-3 of the scale of sub-blocks 4-7; high nibble: bits 0-3 of their minimum | |

**Dequantisation** (`quants.c:1529-1551`): **y = d·sc·q − dmin·m** (`quants.c:1546-1547`), q the 4-bit code.

**Dot product** (`vecdotq.cuh:918-966` and `:508-530`; `vdr` 2, 16 threads per block). Per sub-block, one DP4A chain
for Σ q·q_a and one DP4A of the activations with 0x01010101 for Σ q_a (`vecdotq.cuh:521`). Result
**d · Σ d_a·sc·Σq·q_a − dmin · Σ d_a·m·Σq_a** (`vecdotq.cuh:529`). The multi-token kernel (MMQ) takes the activation
sum from q8_1's `s` field instead (`vecdotq.cuh:552`).

### Q5_K

256 values in 176 B, 5.5 bpw (`block_q5_K`, `common.h:340-356`). Q4_K plus a fifth bit per value.

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | scale of the scales |
| 2 | 2 | `dmin` (f16) | scale of the minimums |
| 4 | 12 | `scales[12]` | as Q4_K |
| 16 | 32 | `qh[32]` | bit k of byte l is bit 4 of value 32k+l |
| 48 | 128 | `qs[128]` | low 4 bits, laid out as Q4_K's `qs` |
| | **176** | | 2 + 2 + 12 + 32 + 128 |

**Dequantisation** (`quants.c:1731-1756`): **y = d·sc·(q4 + 16·h) − dmin·m** (`quants.c:1750`).

**Dot product** (`vecdotq.cuh:968-1017` and `:564-593`; `vdr` 2, 16 threads per block): as Q4_K, with the fifth bit
ORed in as bit 4 of each byte before the DP4A.

### Q6_K

256 values in 210 B, 6.5625 bpw (`block_q6_K`, `common.h:358-368`). Sixteen sub-blocks of 16 values with signed 8-bit
scales; codes are 6-bit, offset by 32. **The f16 scale is last.**

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 128 | `ql[128]` | low 4 bits |
| 128 | 64 | `qh[64]` | high 2 bits |
| 192 | 16 | `scales[16]` (int8) | value v uses `scales[v/16]` |
| 208 | 2 | `d` (f16) | block scale |
| | **210** | | 128 + 64 + 16 + 2 |

The bits of value v, with h = v/128 and l = v mod 32 (`quants.c:1939-1968`):

| v − 128h | low 4 bits | high 2 bits |
| --- | --- | --- |
| l (0-31) | low nibble of `ql[64h+l]` | bits 0-1 of `qh[32h+l]` |
| 32+l | low nibble of `ql[64h+32+l]` | bits 2-3 of `qh[32h+l]` |
| 64+l | high nibble of `ql[64h+l]` | bits 4-5 of `qh[32h+l]` |
| 96+l | high nibble of `ql[64h+32+l]` | bits 6-7 of `qh[32h+l]` |

**Dequantisation:** **y = d · scales[v/16] · (q − 32)** (`quants.c:1953-1957`).

**Dot product** (`vecdotq.cuh:1019-1043` and `:627-647`; `vdr` 1, 32 threads per block): assemble 4 six-bit codes per
word, subtract 32 from each byte (`vecdotq.cuh:641`), DP4A, multiply by the int8 sub-block scale and d_a; result
**d · Σ** (`vecdotq.cuh:646`).

### Q8_0

32 values in 34 B, 8.5 bpw (`block_q8_0`, `common.h:251-256`). One tensor in the file
(`blk.47.ffn_down_shexp.weight`) and the format of the 8-bit KV cache ([ADR-0008](../adr/0008-kv-cache.md)). In
Unsloth's UD-IQ4_XS file ([tensors-UD-IQ4_XS.tsv](../model/tensors-UD-IQ4_XS.tsv)) it is the type of 502 tensors:
the down experts of layers 2, 4, 30, 46 and 47, the dense matrices, the token embedding, and the hyper-connections' and
PLE's matrices.

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | block scale |
| 2 | 32 | `qs[32]` (int8) | value j |
| | **34** | | 2 + 32 |

**Dequantisation** (`quants.c:553-567`): **y = d · q** (`quants.c:564`).

**Quantisation** (what writing the KV cache does): d = max|x| / 127, q = round(x / d), q = 0 when d = 0
(`quantize_row_q8_0_ref`, `quants.c:276-299`; the CUDA KV write, `quantize_f32_q8_0_block`, `cpy-utils.cuh:137-154`,
is the same).

**Dot product** (`vecdotq.cuh:851-866`, `:246-258`; `vdr` 2, 4 threads per block): DP4A straight on the bytes; result
d · d_a · sumi. In attention, llama.cpp's vector flash-attention kernel quantizes the query to q8_1 in shared memory
(`quantize_q8_1_to_shared`, `fattn-common.cuh:332-372`) and dots it with Q8_0 keys by DP4A
(`vec_dot_fattn_vec_KQ_q8_0`, `:304-329`); it dequantizes Q8_0 values to f16 or f32 (`dequantize_V_q8_0`, `:588`). One
K or V row of one head (256 values) is 8 blocks, 272 B.

### Q4_0 and Q5_0

32 values with one f16 scale, 4.5 and 5.5 bpw (`block_q4_0`, `common.h:194-198`; `block_q5_0`, `common.h:230-234`;
`GGML_TYPE_Q4_0 = 2`, `GGML_TYPE_Q5_0 = 6`, `ggml.h:392`, `:396`).

| Type | Offset | Bytes | Field | Meaning |
| --- | ---: | ---: | --- | --- |
| both | 0 | 2 | `d` (f16) | block scale |
| Q5_0 | 2 | 4 | `qh[4]` | bit j: the fifth bit of value j |
| both | 2 (Q4_0), 6 (Q5_0) | 16 | `qs[16]` | byte j: value j (low nibble), value j + 16 (high nibble) |

**Dequantisation** (`quants.c:459-477`, `:500-524`): Q4_0 **y = d · (q − 8)**; Q5_0 **y = d · (q − 16)** with q's
bit 4 from `qh`. Neither has a codebook or a second scale.

### BF16, F16, F32

One value per "block": 2, 2 and 4 bytes, little-endian. BF16 widens to f32 by a 16-bit shift
(`ggml-impl.h:606-613`). These are not dot-producted against q8_1. On the GPU, llama.cpp runs them through its float
kernels (`ggml/src/ggml-cuda/mmvf.cu`) or BLAS. On the CPU the activation is converted to the weight's own type
(`ggml-cpu.c:222`, `:395`). F16 appears once, `blk.1.ple_conv1d.weight` (4 x 10240).

The engine reads some small tensors as BF16 whatever their type in the file: the router, `ssm_alpha` and `ssm_beta`,
the indexer's projections, the hyper-connections' matrices and the PLE's key and value. UD-IQ4_XS stores
some of them as F32 and some as Q8_0. The loader narrows F32 to BF16 by rounding to nearest, ties to even, as the
kernels' `f32_to_bf16` does; on that file every F32 value it narrows has zero low 16 bits, so nothing is rounded. A
quantised tensor is dequantised by ggml's `to_float` and rounded the same way; a Q8_0 value d · q can need 18 bits of
significand, so this rounds.

## The activation formats

### q8_1: the GPU activation format

32 values in 36 B (`block_q8_1`, `common.h:258-269`).

| Offset | Bytes | Field | Meaning |
| ---: | ---: | --- | --- |
| 0 | 2 | `d` (f16) | scale: max\|x\| / 127 |
| 2 | 2 | `s` (f16) | a sum (below) |
| 4 | 32 | `qs[32]` (int8) | round(x / d) |
| | **36** | | 2 + 2 + 32 |

**`s` is not the same in every implementation.** The CPU reference stores d · Σq, the sum of the *quantized* values
(`quantize_row_q8_1_ref`, `quants.c:302-335`). The CUDA kernel that quantizes activations stores Σx, the sum of the
*original* floats (`quantize_q8_1`, `quantize.cu:84-100`); so does Strata's (`include/strata/kernels/native_mmvq.hpp:33-35`
at `1678de3`). None of the file's types reads `s` in the single-token CUDA kernel: Q2_0 folds its offset into the
unpack, and Q4_K and Q5_K compute Σq_a with DP4A. Only the multi-token Q4_K and Q5_K kernels (MMQ) read it
(`vecdotq.cuh:552`, `:615`).

Rounding: CUDA uses `roundf(x / d)` (`quantize.cu:91`), the CPU reference `roundf(x · (1/d))` (`quants.c:323-327`).
The two can differ in the last bit.

### q8_K: the CPU activation format for K- and i-quants

256 values in 292 B (`block_q8_K`, `common.h:370-376`): `d` f32 at 0 (4 B), `qs[256]` int8 at 4, `bsums[16]` int16
at 260 (sums of each 16 codes). The reference quantizer (`quantize_row_q8_K_ref`, `quants.c:2768-2805`) takes the
value of largest magnitude, max, and sets iscale = −127 / max, q = min(127, nearest(iscale · x)), d = 1 / iscale. So
d is negative when the largest-magnitude value is positive, and it keeps f32 precision where q8_1 keeps f16. Strata
uses this rule on its GPU path to match the CPU reference (`include/strata/kernels/quantize_act.hpp` at `1678de3`). The
engine may need it for the same reason.

## The codebooks

All are in `common.h` at `3cf0325`. They are data, and this page does not copy them. The checksum is sha256 over the
values in order, each packed little-endian in the table's C type, so anyone can recompute it from the cited lines.

| Table | Used by | C type x entries | Bytes | Lines in `common.h` | Byte values | sha256 |
| --- | --- | --- | ---: | --- | --- | --- |
| `iq2s_grid` | IQ2_S | `uint64_t` x 1024 | 8,192 | 758-1015 | 8, 25, 43 | `e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88` |
| `iq3xxs_grid` | IQ3_XXS | `uint32_t` x 256 | 1,024 | 1017-1050 | 4, 12, 20, 28, 36, 44, 52, 62 | `46e35f5a997efdee6c99ce57854c8a0d4f0ff8ca57e5e8a60c0793ea580acf5d` |
| `iq3s_grid` | IQ3_S | `uint32_t` x 512 | 2,048 | 1052-1117 | 1, 3, 5, 7, 9, 11, 13, 15 | `bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f` |
| `kvalues_iq4nl` | IQ4_NL, IQ4_XS | `int8_t` x 16 | 16 | 1120-1122 | −127 to 113, increasing | `61aa47540aa024b5d6ddaa839b84ffe59f3d5a349af5c6c7ffcb5e0474b46163` |
| `ksigns_iq2xs` | IQ3_XXS (dequantisation) | `uint8_t` x 128 | 128 | 513-522 | k plus bit 7 for odd parity | `a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31` |
| `kmask_iq2xs` | IQ2_S, IQ3_XXS, IQ3_S (dequantisation) | `uint8_t` x 8 | 8 | 509-511 | the 8 single-bit masks | `5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5` |
| `ksigns64` | IQ3_XXS (SYCL dot product) | `uint64_t` x 128 | 1,024 | 524-557 | 0x00 or 0xFF per byte | `6ca8598509cc63406eaece40cff6b60198f6fa1e0dff261de7bd8d73bfed770c` |

| `iq2xxs_grid` | IQ2_XXS | `uint64_t` x 256 | 2,048 | 560-625 | 8, 25, 43 | `05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54` |
| `iq2xs_grid` | IQ2_XS | `uint64_t` x 512 | 4,096 | 627-756 | 8, 25, 43 | `06e47aaca60b4dc1d9b5a3f34540437058a6b142b4d7a59d5ded769b4d1bf1de` |
| `iq1s_grid` | IQ1_M | `uint64_t` x 2048 | 16,384 | 1135-1648 | −1, 0, 1 | `07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a` |

The IQ3_S file's three grids together are 11,264 B; with the other quants' three the packed buffer the engine uploads
is 33,808 B (`QGRID_BYTES`, shared/quant.h). The sign tables can be replaced by arithmetic (parity and shifts), as the
CUDA kernels do. `kvalues_fp4` belongs to a type none of the files holds.

## Wide loads: a layout with quants in one run and scales in another

[ADR-0018](../adr/0018-one-file-iq3s.md) (decision 3) says each expert is stored for wide, contiguous loads, whatever
the file's layout. This section says what that means per type and what it costs. On the B60 the runs layout is
3.2-8.8 times faster than the file's layout for every type tried: IQ2_S 3.6x, IQ3_S 4.8x, IQ4_NL 3.2x, Q6_K 8.8x
(measured with prototype kernels; machine, versions and method in the [kernel record](../measurements/MEASUREMENTS.md#2026-10-11-kernel-spike)).

**The problem in the file's layout.** A block's fields are interleaved (array of structures), and six of the types
have a block size that is 2 mod 4. Their blocks, and the quants inside them, start only on 2-byte boundaries. For this
reason the CUDA kernels read those types 16 bits at a time (`get_int_b2`, `vecdotq.cuh:18-25`) and only the 4-byte-aligned
ones as words (`get_int_b4`, `:27-29`). A 16- or 32-byte load of quants crosses into the next block's scale.

| Type | Bytes / block | Block start alignment in a row | Row of the file's experts | Row bytes | Row bytes mod 16 |
| --- | ---: | ---: | --- | ---: | ---: |
| IQ2_S | 82 | 2 | 2,560 values, 10 blocks | 820 | 4 |
| IQ3_XXS | 98 | 2 | 10 blocks | 980 | 4 |
| IQ3_S | 110 | 2 | 10 blocks | 1,100 | 12 |
| IQ4_XS | 136 | 8 | 10 blocks | 1,360 | 0 |
| IQ4_NL | 18 | 2 | 640 values, 20 blocks | 360 | 8 |
| Q2_0 | 18 | 2 | 640 values, 10 blocks | 180 | 4 |
| Q8_0 | 34 | 2 | 640 values, 20 blocks | 680 | 8 |
| Q4_K / Q5_K / Q6_K | 144 / 176 / 210 | 16 / 16 / 2 | (dense) | | |

**llama.cpp's SYCL "reorder".** At first use, SYCL rewrites a tensor in place as separate runs: all blocks' quants,
then all scales (`reorder_qw`, `ggml-sycl.cpp:4619-4659`). For Q4_0 it is `qs` then `d` (`:4163-4200`); for Q8_0 the
same (`:4202-4239`); for Q4_K `qs`, then `scales`, then `dm` (`:4241-4283`); for Q6_K `ql`, `qh`, `scales`, `d`
(`:4566-4617`). For MoE tensors it reorders **each expert's slice separately**, so that one expert stays one contiguous
byte range (`:4285-4327` for Q4_K; Q5_K and Q6_K at `:4330`, `:4379`). Each reorder copies the tensor to a temporary
device buffer of the same size and scatters it back with a kernel (`:4163-4176`). The MoE reorder takes only Q4_K,
Q5_K and Q6_K (`opt_for_reorder_id`, `:4705-4719`); no type the file's experts use is reordered.

**The runs layout for the file's expert types.** Each expert slice is stored as runs in the order below: quants
first, the largest run first, the f16 scales last. The byte counts are per expert: gate or up is 640 rows x 2,560
values (6,400 super-blocks), down is 2,560 rows x 640 values.

| Type | Runs, bytes per block | Run bytes per expert slice | Slice |
| --- | --- | --- | ---: |
| IQ2_S | grid index low bits 32, signs 32, `qh` 8, `scales` 8, `d` 2 | 204,800 + 204,800 + 51,200 + 51,200 + 12,800 | 524,800 |
| IQ3_XXS | grid indices 64, scale-and-sign words 32, `d` 2 | 409,600 + 204,800 + 12,800 | 627,200 |
| IQ3_S | `qs` 64, `signs` 32, `qh` 8, `scales` 4, `d` 2 | 409,600 + 204,800 + 51,200 + 25,600 + 12,800 | 704,000 |
| IQ4_XS | `qs` 128, `scales_l` 4, `scales_h` 2, `d` 2 | 819,200 + 25,600 + 12,800 + 12,800 | 870,400 |
| IQ4_NL (down) | `qs` 16, `d` 2 | 819,200 + 102,400 | 921,600 |
| Q2_0 (down) | `qs` 16, `d` 2 | 409,600 + 51,200 | 460,800 |
| Q8_0 (down) | `qs` 32, `d` 2 | 1,638,400 + 102,400 | 1,740,800 |

Every run length is a multiple of 256 B, so if a slice starts on a 256-B boundary every run does, with no padding.
Within a run, block k's piece sits at k x its size: the 64- and 128-byte quant pieces are 64- and 128-byte aligned, and
a row's quants are one contiguous stretch (10 or 20 pieces).

```
file (one expert slice):   [d|qs|qh|signs|sc][d|qs|qh|signs|sc] ... 6,400 blocks of 110 B      (IQ3_S)
reordered (same bytes):    [qs x 6400 ][signs x 6400][qh x 6400][sc x 6400][d x 6400]
                            409,600 B    204,800 B     51,200 B   25,600 B   12,800 B
```

**What the engine does.** The reorder is a permutation of each slice, so it costs no bytes. The host arena keeps the
file's layout, and the CPU computes the misses from it with ggml's kernels (`core/cpu/experts.h`). Each card's cached
experts are scattered into the runs layout on the host as their slots are filled, at load and on every swap
(`core/load/runs.cpp`, called from `model.cpp` and `swap.cpp`). The prompt path reads the experts it streams from the
arena in the file's layout (`qgemm_<type>_file`, `kernels/quant/decode.h`).

Strata's CUDA kernels do not reorder: they read the GGUF's blocks unmodified, row-major
(`include/strata/kernels/native_mmvq.hpp:20-23` at `1678de3`).

## Sources and how this was checked

- **Code read:** the files and lines cited in each section, at `3cf0325`, and Strata's `include/strata/kernels/`
  headers at `1678de3`.
- **Block sizes:** each struct's size, added up field by field, matches its `static_assert` in `common.h`. A scratch
  script (not committed) takes every row of [tensors-IQ3_S.tsv](../model/tensors-IQ3_S.tsv), multiplies out the
  shape and checks bytes = values / values-per-block x bytes-per-block with the table at the top. All 1,224 tensors
  match, and every row length (the first dimension) is a whole number of blocks. The row lengths are 2,560 (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS), 160 and 640 (IQ4_NL), 640 (Q2_0, Q8_0), 2,560 and 6,144 (Q4_K, Q5_K,
  Q6_K).
- **Codebook checksums:** a scratch script parses each table between its `GGML_TABLE_BEGIN` and `GGML_TABLE_END`
  lines, packs the values little-endian in the table's C type and takes the sha256. It also lists the distinct byte
  values of the grid tables.
- **Tests on the B60:** `core/tests/matvec_test.cpp`, `moe_test.cpp` and `qgemm_test.cpp` check every type against
  ggml's dequantisation.
