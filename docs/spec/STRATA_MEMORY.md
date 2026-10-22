# Strata's design, part 2: the memory system and the prompt path

Where Strata keeps the experts, the KV cache and the n-gram table, and how it reads a prompt: what each mechanism
does, its data and sizes, its files and settings, and what Strata measured it to be worth. Strata is the design
reference ([ADR-0001](../adr/0001-level-zero-from-scratch.md)), so its mechanisms are written down as behaviour. No
Strata code is copied.

**Source.** [avra-m3/strata-intel](https://github.com/avra-m3/strata-intel) `main` at commit `1678de3`
(2026-10-02), the same commit as [part 1](STRATA_DECODE.md). Paths below are relative to that tree.
Strata's own docs are cited as *DETAILS* (`docs/DETAILS.md`), *MULTI_GPU* (`docs/MULTI_GPU.md`), *UNSLOTH*
(`docs/UNSLOTH_Q4.md`) and *the paper* (`docs/paper/Strata-Paper.pdf`). Run results are under `bench/results/`.

**Where the numbers come from.** Unless a line says otherwise, every measured number is Strata's own, taken on its
reference PC ([part 1](STRATA_DECODE.md) describes it). Most were measured on Q2_0, IQ2_XS or IQ3_XXS, not
on IQ3_S, the one file this project runs ([ADR-0018](../adr/0018-one-file-iq3s.md)). Where Strata has an IQ3_S
number, it is given.

**Engine default and setup default.** The engine binary's defaults (`src/program/generate.cpp`, `struct Options`)
are not always what Strata runs with. `setup.py` writes the run configuration (around line 3124). Both are given where
they differ.

## Contents

1. [The expert arena](#1-the-expert-arena)
2. [The VRAM expert cache](#2-the-vram-expert-cache)
3. [The CPU pool](#3-the-cpu-pool)
4. [The PCIe share](#4-the-pcie-share)
5. [The KV cache](#5-the-kv-cache)
6. [The layer split](#6-the-layer-split)
7. [The prompt path](#7-the-prompt-path)
8. [Conversation checkpoints](#8-conversation-checkpoints)
9. [The n-gram reader](#9-the-n-gram-reader)
10. [Sizes checked against Strata's docs](#10-sizes-checked-against-stratas-docs)

## 1. The expert arena

**What it does.** By default, every routed expert of the model (48 layers × 512 = 24,576) is copied into host RAM once
at start. The arena is one allocation, page-locked where the driver allows it. The CPU computes missed experts in place
from the arena (section 3). The GPU fills its cache from the arena and reads the PCIe share of the misses from it
(sections 2 and 4). The arena is the expert tier between the SSD and VRAM.

**Two layouts ("packs").** An expert is one *blob*. A pack's `experts.bin` holds the blobs layer-major, expert-minor:
blob (l, e) sits at the layer's offset plus `e × blob_bytes(l)`, with no padding. `ExpertLayout`
(`include/strata/kernels/cpu/expert_layout.hpp`) is the single table that every consumer asks for offsets: the arena,
the VRAM cache, the prompt path, the CPU pool and the GPU window.

| Pack | Built by | Blob | Size of the arena |
| --- | --- | --- | --- |
| **Canonical** (Q2_0 only) | `tools/strata_pack.py` (+ `pack_layer.py`) | Strata's own planar form. Gate+up codes, then down codes, then gate+up FP16 scales, then down scales. All three projections Q2_0, 64 weights per scale. **1,382,400 B**, the same for every layer | `experts.bin` = 33,973,862,400 B (24,576 × 1,382,400). Setup writes this ~40 GB copy only for Q2_0 on an AVX-512 CPU (DETAILS, "Before you start") |
| **Native** (any GGUF, the IQ files) | `tools/iq_pack.py` | The three GGUF slices of one expert, back to back: gate rows, then up rows, then down rows, in the file's own quant types. Size is **per layer**, because RCO picks a type per tensor | Q2_0 34.0 GB, IQ2_XS 35.5 GB, IQ3_XXS 42.9 GB, **IQ3_S 50.3 GB**, Coder 23.4 GB (`setup.py` `MODELS`, `arena_gb`) |

A native pack has `native_experts.txt`. It has one line per layer: the gate/up type, the down type, the layer's offset,
the blob size, the offsets of the three roles inside the blob, and (v3/v4) which GGUF shard holds each role. With
`experts.bin` absent, the engine reads that layer's tensors from the GGUF itself, at offsets recorded per layer and
role. Before it allocates anything, it checks each recorded span against the GGUF's tensor name, type, shape and
bounds (`check_experts_gguf`, `src/core/expert_source.cpp`).

**How it is filled** (`ArenaExpertSource::open`, `src/core/expert_source.cpp`; `load_experts_ranges`,
`src/core/pinned.cu`):

- The file size is checked against the layout before the allocation.
- **One registration slice per layer.** Every expert then lies inside one slice, so a DMA never spans two
  registrations. The arena is one blob longer than the data, so a whole-slot copy can start at any expert.
- Reader threads each open their own handle. From `experts.bin` they read 8 MiB chunks. From the GGUF they read 16
  experts per role per read and scatter them into the blob layout. Each layer gets an FNV-1a checksum. Plain `fread`
  is used: MSVC's `ifstream` split reads into 4 KiB calls and fell to 0.03 GiB/s.
- **Page backing.** Strata tries 2 MB large pages first (`MAP_HUGETLB` / `MEM_LARGE_PAGES`), then normal pages. It
  then registers the arena with CUDA (`cudaHostRegister`, mapped). If the whole range is refused, it registers per
  layer, and any rest is locked in the working set (`include/strata/core/pinned.hpp`). Under WDDM only 8 GiB is
  registered by default. Linux registers the whole arena. `STRATA_ARENA_PIN_GIB=N` caps it (MULTI_GPU, "Limits").
- Each registered slice gets a device alias (`cudaHostGetDevicePointer`). That is what lets the GPU read a missed
  expert over PCIe (section 4).

**Why a copy and not a mapping.** The arena exists because mapping `experts.bin` was slow. The 34 GB of experts plus
the 26.8 GiB n-gram mapping were 60.8 GB of file-backed pages on a 63 GB machine. The OS dropped and re-faulted
expert pages. The pool then ran at ~19 GB/s in the engine against 42.8 GB/s in a micro-benchmark reading anonymous
memory. That is the "2.2x" recorded in `include/strata/core/expert_source.hpp` (the comment above
`ArenaExpertSource`).

**Low-RAM variants** (DETAILS, "Low-RAM mode"; `FileExpertSource`, same files):

| Setting | What it does | Measured |
| --- | --- | --- |
| `--mmap-experts` | No arena. The experts are mapped from the pack's `experts.bin`, or (0.1.31, native packs) from the GGUF shards in place. The OS file cache holds what the GPU does not. An expert read from the GGUF is three reads, fetched for a layer on `STRATA_FETCH_THREADS` (8) threads. On Windows they go out as one `PrefetchVirtualMemory` batch | Coder: engine's committed memory 36 to ~13 GB, same answers. With a small GPU "much slower" (no number given) |
| `--resident-experts` | Only the experts that no VRAM slot holds (the "complement") are copied into RAM at start: page-locked if possible, otherwise locked. Strata keeps 4 GiB of RAM free (`STRATA_RESIDENT_HEADROOM_GIB`). If the complement does not fit, it falls back to the mapped mode. Adaptive swaps exchange bytes between a VRAM slot and the RAM copy, so RAM always holds exactly what the GPU lacks | 0 file reads in steady use (server log `blob reads from the file`). No speed figure |
| `--resident-budget-gib N` | The N GiB of complement experts that the profile ranks hottest are copied into RAM. The rest are read from the files. Adds the routing lookahead ([section 2](#routerlookahead)) | UD-Q4_K_XL (72 GiB of experts) on 64 GB: 7-8.5 tok/s at N = 40, against ~3 before (DETAILS); 5.1-5.2 at N = 24 (UNSLOTH) |
| `--shared-expert-arena FILE` | Linux: the arena is backed by a `MAP_SHARED` file, so several processes can share it | Not measured |

**Settings:** `--pack`, `--native`, `--mmap-experts`, `--resident-experts`, `--resident-cpu-experts`,
`--resident-budget-gib`, `--shared-expert-arena`, `STRATA_ARENA_PIN_GIB`, `STRATA_RESIDENT_PIN`,
`STRATA_RESIDENT_HEADROOM_GIB`, `STRATA_FETCH_THREADS`, `STRATA_NO_LARGEPAGES`.

**Measured effect.**

- Arena against mapping: 2.2x on the pool's bandwidth (above).
- Load: 1.42-1.52 GiB/s, ~35 s for ~40 GB, from a terminal. Under Task Scheduler's defaults: 0.05 GiB/s (DETAILS,
  "Running it at startup"). The paper (Finding 13) adds that on Windows, page-locking 34-43 GB as one range fails, and
  per-layer ranges are what work.

## 2. The VRAM expert cache

**What it does.** VRAM that is left after the dense weights, the session (KV and recurrent state), the draft layer and
a reserve becomes *slots*. Each slot holds one expert blob. A residency table maps (layer, expert) to a slot or -1. The
GPU computes the routed experts it holds; the CPU and the PCIe share handle the rest.

**Data and sizes** (`include/strata/core/expert_cache.hpp`, `src/core/expert_cache.cpp`; sizing in
`src/program/generate.cpp` around line 2664):

- **`--expert-cache auto`** (setup's default): slots = (free VRAM − reserve) / largest blob. The reserve is
  `--vram-reserve-mib` (700 MiB by default; setup raises it for the image encoder), plus the draft head's bind bytes,
  plus the prompt buffers when they are not borrowed. With a profile, slots ≤ the profile's length.
- **Slots sized per blob** (native packs with a profile): the slots are cut to each profiled pair's own blob size,
  rounded up to 256 B, in profile order, until the budget is used. The paper (Finding 8) measured this on IQ3_XXS:
  blobs are 1.3-2.3 MB, and sized slots fit 3,556 experts instead of 2,673, for +13% decode at 1K.
- After the slots are written, free VRAM is read again. If the reserve is no longer free, the cache is reopened
  smaller, because WDDM does not commit memory until it is touched.
- Every slot is filled at start and checked: slot 0 is read back and compared byte for byte (`verify_slot`).
- Experts per GB: ~700 for the 2-bit files (DETAILS). For IQ3_S's ~2.05 MB average blob, about 490 per GB
  (*estimate*: 50.3 GB / 24,576 blobs, then 1 GB / 2.05 MB).
- Experts that fitted on the 12 GB card (paper, Table 4): Q2_0 4,517 at 1K and 1,588 at 262K; IQ3_XXS 3,556 at 1K
  and 2,407 at 128K.

**The profile fills it hottest first.** The cache takes the profile's ranked (layer, expert) pairs in order until its
slots are full.

### The shipped profile, `data/expert-profile.bin`

| | |
| --- | --- |
| **Format** | Little-endian. `STRP` magic, then five u32: version (1), n_layers (48), n_expert (512), slots, n_ranked. Then n_ranked pairs `(u16 layer, u16 expert)`, hottest first. Then an int32 table of n_layers × n_expert giving each pair's rank (its slot when the cache is that big; -1 if it is not ranked). Written and read by `tools/make_profile.py`; read by `read_expert_profile` (`expert_cache.hpp`). That header calls the trailing table "a frequency table". In the shipped files it holds the rank index, checked by decoding them |
| **Shipped file** | 196,632 B. slots = n_ranked = 24,576, so it ranks every pair. The top 4,500 pairs give each layer 73-114 experts; the top 8,000 give 118-217. The Coder's `expert-profile-coder.bin` (98,328 B, 48 × 256) is this ranking mapped through the Coder release's `rco-allocation.txt`, with pruned experts dropped (`bench/results/2026-09-28-coder/README.md`) |
| **How it was made** | Two commits build it. Commit `f2a08d4` (2026-09-24) ships 8,000 ranked pairs, built for 8,000 slots. Commit `e6265c7` (2026-09-28, issue #46) keeps those 8,000 and adds the other 16,576: first pairs from routing traces, ranked by frequency, then every pair still missing, interleaved across layers (`make_profile.py`). **The repository does not record which prompts and which model file produced the first 8,000.** The engine header quotes a ten-fold leave-one-out from `tools/routing_kfold.py`, which is not shipped. The paper's Figure 3 scores "each prompt against a profile built from the other nine", which points to ten prompts. A run record names a `profile-decode-8k.bin` on Q2_0 (`bench/results/2026-09-27-cache-parity`). The traces behind the second commit are not shipped either. A user can rebuild a profile with `--dump-routing` and `make_profile.py` |
| **Licence** | No separate licence. It is part of the strata-intel repository, under its MIT licence (`LICENSE`, "Niko1221 and the Strata contributors"). The only file under `data/` with its own licence is the speed-projection vector (Qwen Community License 1.0) |
| **Matches the GSQ-RCO files?** | **By index, yes; by ranking, not measured for IQ3_S.** The profile holds (layer, expert) ids of the 48 × 512 router of Qwen3.8-Flash-Next. All four GSQ-RCO sizes keep all 512 experts per layer and their router, so the ids name the same experts in IQ3_S. Setup passes this one profile to every size of the original model, IQ3_S included (`setup.py` line 3125). But the profile was probably recorded on Q2_0 (see above), and no Strata record scores it against IQ3_S's own routing. The [expert hit-rate record](../measurements/MEASUREMENTS.md#2026-10-12-expert-hit-rates) measures that on single-sequence decode routing traces of the IQ3_S file. The profile's order also counts experts, not bytes, while IQ3_S's blob size varies by layer |

### Adaptive swaps (`--adapt-every`, `--adapt-swaps`)

This is decode only (`generate.cpp` around line 4459; the same rule in the one-shot loop around line 6048):

- Every routed (token, expert) entry of a verify window adds 1 to a per-(layer, expert) usage count
  (`ExpertDispatch::usage`, `expert_pool_dispatch_multi`).
- Every `adapt_every` rounds (default **4**), for each layer:
  - candidates are missed experts with usage ≥ 2, highest first;
  - victims are the layer's resident experts, lowest usage first;
  - pairs are taken while candidate ≥ victim + 1.5.
- Across layers, the pairs are sorted by gain and cut to `adapt_swaps` (default **96**). Then every count is
  multiplied by 0.7.
- A swap stays in its layer's slot. That matters for sized slots and for a layer split.
- **Non-blocking:** the victim is evicted at once (its residency goes to -1, so the CPU computes it). The new
  expert's copy goes on a separate stream. The table entry is set and uploaded only when the copy's event has
  completed, and the next round never waits for it.
- With `--resident-experts`, the evicted blob is copied back into the RAM complement (`stage_exchange` /
  `commit_exchanges`).
- `--adapt-every 100000` (or `--adapt-swaps 0`) makes residency static. DETAILS recommends it for reproducible sampled
  output.

### Hit rates measured

| What | Hit rate | Source |
| --- | --- | --- |
| Profile only, ~4,500 slots, held-out prompts | 0.50 | Paper, Figure 3 and Finding 4 |
| Profile, ten-fold leave-one-out, 4,105 slots | 0.6447 [0.611, 0.675] | `expert_cache.hpp` header (the tool is not shipped) |
| Profile + adaptive swaps, Q2_0 4K, 12 GB | 0.72 | Paper, Table 5 |
| IQ2_XS / IQ3_XXS, the same | 0.78 / 0.71 | Paper, Table 5 |
| Coder, 12 GB card, 2,537 slots, 4K code prompt | 0.72 (a wrong index mapping would give ~0.21) | `bench/results/2026-09-28-coder` |
| Compulsory-miss fill instead of a profile | 0.4864 | `generate.cpp`, `Options::expert_profile` |
| One global slot counter (arrival order), 256 slots | 0.0297; 8 per layer would give 0.214 | `expert_cache.hpp` (Strata's R4.2g) |
| All 10 of a layer's experts resident, held-out | 0.0456 of (layer, token) pairs | `expert_cache.hpp` header |

Effects on speed:

- Making refills non-blocking took Q2_0 at 4K from 91.7 to 94.4 tok/s (paper, Finding 10).
- Turning the cache on changes the top-1 token at 2-5% of positions (near-ties) with no change in perplexity
  (`bench/results/2026-09-27-cache-parity`).

### RouterLookahead

Declared in `include/strata/core/expert_source.hpp`, implemented in `src/core/expert_source.cpp`, and started in
`generate.cpp` around line 3026.

- **What it computes.** When the CPU pool starts layer l, a background thread multiplies layer l+1's router (a host
  copy in BF16, 512 × 2,560 per layer, ~126 MB for all layers) by layer l's MoE input (up to 8 tokens). For each token
  it keeps the top `STRATA_LOOKAHEAD_K` (default 10) experts. It drops those the GPU cache holds and those already in
  the RAM copy, and calls `warm` for the rest. This works because the residual stream changes little from one layer
  to the next.
- **It only warms SSD pages.**
  - `warm` asks the OS for the predicted experts' pages (`madvise(WILLNEED)`, or `PrefetchVirtualMemory` on Windows).
  - It never changes which experts are computed, and never copies anything to the GPU.
  - It runs only when the expert source reads files in place (`warms()`: GGUF mapped, with `--resident-budget-gib` or
    `--mmap-experts` on a native pack). With the arena it is off.
  - A submit that arrives while a prediction is still running is skipped.
  - `STRATA_LOOKAHEAD=0` turns it off.
- **Measured.** UNSLOTH: with UD-Q4_K_XL read from the file tier, "about half of the SSD reads were predicted", and
  decode went from 6.9 to 7.9 tok/s (+14%). That is recall among the coldest experts, at k = 10. DETAILS' other
  figure is not isolated: 5.1-5.2 tok/s (N = 24, "before the routing prefetch") to 7-8.5 (N = 40), with the RAM
  budget changed in the same step. `--stats` reports how many file reads had been warmed, but no
  recall figure is published.

## 3. The CPU pool

**What it does.** For each layer of a verify window, it computes the distinct missed experts that are not given to the
PCIe share, in place from the arena, while the GPU computes the resident ones (`include/strata/kernels/cpu/pool.hpp`,
`src/kernels/cpu/pool.cpp`; dispatch in `expert_pool_dispatch_multi`, `src/core/expert_source.cpp`).

**Workers and pinning.**

- By default there is one worker per physical core except the first, which is reserved for the host loop
  (`physical_cores(skip_first)`).
- Each worker is pinned to one logical CPU per physical core, never to an SMT sibling of another worker.
- The host thread is pinned to the reserved core (`pin_current_thread`).
  - Measured: 36.32 GB/s with 5 workers alone, and 26.9 GB/s inside the host loop when the host was unpinned.
- The host also drains jobs during `run()` (`host_works`, on by default), so six threads work on six cores.
  - Measured: 33.7 GB/s against 36.8 expected for 5 workers. That gap is the reason it is on.
- `--pool-workers N` overrides the count. `--pool-affinity auto|p-cores|all` (default `all`) is for hybrid CPUs
  (#272). Setup's `--calibrate` tries worker counts.
- Parked workers spin for 20 ms (`kSpinBeforeSleep`; `STRATA_POOL_SPIN_US`) and then sleep on a condition variable.
  This fixes 100% CPU when idle (issue #4).
- Claims carry an epoch in one 64-bit word, so a late worker cannot take a job from the next batch (issue #29). A wait
  with no progress for 60 s stops the engine (`kStall`). Each hot atomic has its own cache line.
- Every expert is split by rows across all threads (`run_split_multi`): gate/up rows, then quantize the intermediate,
  then down rows. That way, a few misses still use every core. `--no-split-rows` is the A/B switch.

**Jobs.** Experts are grouped per window: one distinct expert computes all tokens routed to it (up to `MAXT` = 8).
The scratch memory is owned per worker, and nothing is allocated on the token path.

**Kernels.**

| Pack | Kernel | Speed (6 cores, Ryzen 5 7600, DDR5-5200) |
| --- | --- | --- |
| Canonical Q2_0 | Strata's AVX-512 kernel (VNNI `vpdpbusd`, VBMI `vpmultishiftqb` unpack; the activation is Q8 in 32-element chunks with its own scales). AVX-512 is required | 42.55 GB/s, DRAM-bound (`include/strata/kernels/cpu/expert.hpp`). About 42 GB/s at decode next to the GPU; RAM gives 41 GB/s concurrent, 52 alone (paper, Finding 6) |
| Native, one token | ggml-cpu's own `vec_dot` for the layer's types (the activation quantized to the type's `vec_dot_type`) | i-quants ~5 GB/s per core, 23-26 GB/s on six: **arithmetic-bound** (codebook lookups, signs), not RAM-bound (paper, Finding 7) |
| Native, ≥ 2 tokens on one expert | Strata's multi-token kernels (`src/kernels/cpu/iq_avx512.cpp`, an AVX-2 variant, `iq4nl256_down_rows`). They decode 64 weights once and apply them to every token. Formats: IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S for gate/up; IQ4_NL for down (AVX-2). **IQ4_XS gate/up (IQ3_S's layer 47) stays on ggml** | **2.0-2.4x ggml at three tokens, no faster at one** (`src/kernels/cpu/native_expert.cpp`, measured by `native_expert_parity`; paper, Finding 7). Most experts serve one token |
| Native Q2_0 down rows | Strata's own SIMD kernel (ggml has no x86 SIMD for Q2_0) | 7x the scalar kernel (paper, Finding 7) |

**Arithmetic-bound speed per core:** ~5 GB/s of i-quant weights per core, with both ggml's AVX2 kernels and Strata's
AVX-512 ones (paper, Finding 7). On IQ3_S decode the gate/up rows run at ~30 GB/s on 6 cores
(`iq_avx512.cpp`, the prefetch comment).

**Settings:** `--pool-workers`, `--pool-affinity`, `--no-host-worker`, `--no-split-rows`, `--no-pool` (the GPU-only
floor), `STRATA_IQ_MT_MIN` (default 2; 1 = the multi-token kernels for every group, so output does not depend on the
drafting, at −1..−3% IQ3_S decode), `STRATA_NO_IQ512`, `STRATA_NO_IQ256`, `STRATA_NO_IQ4NL`, `STRATA_IQ_PREFETCH`
(2,048 B), `STRATA_IQ_GATHER` (opt-in, 3-5% slower on Zen 4), `STRATA_FORCE_AVX2`.

**Measured share of a round** (paper, Table 5, 4K): the GPU waits on the CPU for 13.4 ms of a 34.3 ms window on
Q2_0, 21.1 of 42.0 on IQ2_XS, and 27.1 of 49.4 on IQ3_XXS.

## 4. The PCIe share

**What it does.** For each layer of a verify window, a share of the distinct missed experts is read by the GPU from
the pinned arena over PCIe instead of being computed by the CPU (`expert_pool_dispatch_multi`).

**How misses are split.** The CPU thread plans the layer as soon as the doorbell gives it the ids:

1. It walks the distinct experts in routing order and counts the misses (`nmiss`).
2. `pcie_num` = round(`pcie_frac` × 256). The GPU takes m = ⌊`nmiss` × `pcie_num` / 256⌋. These are the **last m
   distinct misses in routing order**, up to the staging capacity and at most 64. An expert qualifies only if its
   blob is pinned (registered).
3. The plan is published to mapped host memory before the CPU starts its own share. It holds the VRAM groups, then
   the PCIe groups, each with destination rows. The GPU computes its cached experts first and the copied ones as they
   land.

**How the copy is made** (`--pcie-mode`, `GpuPlanSink::pcie_mode`):

| Mode | How | Status |
| --- | --- | --- |
| `dma` (0) | The CPU thread issues copy-engine DMAs into staging slots when it plans the layer. The GPU waits on a flag | The paper's version (Finding 9). It hung inside the driver (#31). Kept as an option |
| `direct` (1) | The grouped kernel reads the mapped arena directly | Option |
| `kernel` (2) = **`auto`** | A copy kernel inside the graph stages the blobs | The default for every pack since `e6265c7` (no host CUDA call in the window). IQ3_S decode 45.3 to 44.8 tok/s against DMA |

**Default share** (`generate.cpp` around line 1721): 0.2 for the canonical Q2_0 pack. For native packs, 0.55 scaled
by a host-to-device probe:

- ≥ 20 GB/s: 0.55;
- below 4 GB/s: 0;
- otherwise min(0.55, max(0.05, 0.55 × bw/26)).

A request may override it (`strata_tune`). On a layer split, each stage probes its own link.

**How the share is measured.** `--calibrate` (`tools/calibrate.py`) sweeps `pcie_frac` ∈ {0, 0.2, 0.35, 0.55, 0.75}
plus the current default. It uses three fixed prompts and 128 tokens each, measuring decode only. Then it re-measures
the best against the default, interleaved, and keeps it only if it is more than 3% faster.

**Measured effect** (paper, Finding 9):

- At 55% by DMA: IQ2_XS at 4K went from 70 to 78 tok/s, and IQ3_XXS from 56 to 65.
- As a copy kernel on the GPU's critical path, more than 20% made decode slower.
- Q2_0 stays at 20%, because its CPU already uses the RAM bandwidth.

`--pcie-frac 0` makes all misses CPU work. That makes output reproducible, at a decode cost (DETAILS).

## 5. The KV cache

**Layout** (`include/strata/core/layer.hpp` `QsaState`, `src/core/layer.cpp`). Only the 12 QSA layers and the MTP draft
layer have a KV cache. The 36 GDN layers keep a fixed-size state.

- KV is stored in pages of 4 cells (one indexer block), all KV heads, K and V.
- A page table maps each logical page to a physical one.
- Bytes per cell per QSA layer:

  | `--kv` | Bytes per cell |
  | --- | ---: |
  | `fp16` | 2,048 |
  | `int8` (one FP16 scale per 64 values) | 1,056 |
  | `q4_0` (Hadamard-rotated) | 576 |
  | `k8v4` | 816 |

- Besides KV, each QSA layer keeps:
  - its pooled indexer keys: one FP32 key of 128 per 4 cells, 128 B per cell, always in VRAM;
  - an indexer tail;
  - a RoPE table shared across the layers: 256 B per cell, 64 MiB at 262K.

**Defaults.** The engine's default is `fp16`. Setup writes `int8` above 8,192 tokens of context, `fp16` up to 8K
(`setup.py` line 2900). DETAILS says "8-bit KV above 4K"; the code says above 8K. The paper says "past 8K".

**KV streaming (`--kv-resident N`).**

- **What stays resident.** Each streamed QSA layer keeps its authoritative K/V in pinned, device-mapped host memory,
  in the identity layout. VRAM holds `n_slots` pages, with `n_slots` = max(N, 20,480) / 4. The minimum is one
  window's selections: 8 queries × 2,051 cells. The page table becomes a residency map.
- **How cells are chosen** (`include/strata/kernels/kv_stream.hpp`):
  - After the indexer selects its ≤ 2,048 positions, a device kernel walks the selected blocks.
  - A resident block gets its reference bit set.
  - A missing block takes a victim slot by a **CLOCK (second-chance) sweep** that never evicts a block the same call
    uses.
  - A copy kernel then reads the missed blocks from host memory over PCIe (zero-copy).
  - All of this happens inside the captured graphs.
- **Writes.** Appends always write the host copy, and write the slot only if the block is resident. Slots never go
  stale, so nothing needs invalidating after a rejected draft or a resume.
- **The draft layer** uses a ring of its window (`--mtp-window`, 32,768 cells) at `block % n_slots`, refilled from the
  host copy on a resume.
- **Setup.** Setup writes `--kv-resident 32768` from 64K context, when RAM has room. It is never on with `k8v4` (the
  engine refuses) or under WSL. RAM cost: 13 × 1,056 B = **~13.7 KB per context token** (12 QSA layers + the draft
  layer, int8), 1.7 GB at 128K.

**Measured gain** (DETAILS, "KV streaming"; engine 0.1.5):

- Q2_0 at 262K: **50.9 to 62.6 tok/s**, with 1,589 to 3,872 experts in VRAM.
- At 128K: about +6%.
- The attention reads the same values (only where they live changes).
- Setup's comment says +23% at 262K. That matches 50.9 to 62.6.

Related measured options:

- `--kv q4_0`: about 4% faster at 128K, perplexity +8-12% (`bench/results/2026-09-27-kv-q4`).
- `--kv k8v4`: 23% less KV than int8. RTX 3090 Coder at 198K: 85 to 99 tok/s.

## 6. The layer split

**What it does.** Layers are cut into contiguous ranges, one per GPU (`--layer-split K1[,K2]|auto`,
`--split-device`). This is pipeline parallelism, not tensor parallelism. Each card runs its range of every verify
window. The last card also runs the output head and the draft layer (MULTI_GPU).

**What each card holds** (MULTI_GPU, "What each card holds"):

- Every card has:
  - a copy of the dense weights (~3.4 GB for the Coder);
  - the session state for **its own layer range only** (the carve, `session_bytes(g, ctx, K, lb, le)`);
  - its verify window (~75 MiB measured, 96 MiB reserved);
  - its prompt buffers;
  - an expert cache for its own layers, filled from the profile.
- The last card also holds the head and the drafter (~0.8 GB; 1,000 MiB reserved).
- Host RAM holds the expert arena once, shared by all cards. The CPU pool computes whatever no card holds.

**The hand-off.**

- **Once per window, not per layer.** The residual leaves one stage and enters the next through mapped pinned host
  memory (`cudaHostAllocMapped | Portable`). No peer-to-peer is used.
- Size: (hc × n_embd + n_embd + hc) floats per token = (4 × 2,560 + 2,560 + 4) × 4 = 51,216 B, × up to 8 window
  tokens ≈ 400 KB (`Verifier::handoff_floats`).
- In the prompt path, stage i reads chunk c+1 while stage i+1 reads chunk c (`Prefill::set_stage`).

**How `auto` prices cache coverage** (`generate.cpp` around line 2261). MULTI_GPU describes `auto` as "keeps the
placement whose caches hold the most of the profile". **The code at this commit uses a cost model instead**:

- Predicted window time = Σ over stages of (layers × per-layer ms) + 190 ms × (1 − routed mass the caches hold).
- Per-layer ms = 0.33 × (84 SMs × 2.617 GHz) / (the card's SMs × clock).
- Routed mass of profile rank r = (r+1)^−1.2, normalised.
- Each stage's cache is filled with its layers' profiled pairs, hottest first, until its free VRAM, after its session
  carve and reserves, is used.
- Every placement is tried for 2-3 cards. With more cards, layers are shared in proportion to speed.
- The constants were fitted on the 5080 + 3090 sweep: K = 24/26/28 predicted hit 99.53/99.34/99.15%, and measured
  99.5/99.4/99.0%. `STRATA_SPLIT_MISS_MS` tunes the miss cost.

`--split-skip-if-fits` runs on the first card alone when it holds every profiled pair, the KV, the drafter and the
reserve.

**Measured** (MULTI_GPU; `bench/results/2026-09-29-layer-split`; Coder, RTX 5080 + RTX 3090, 32K context):

| | Prompt 16K / 28K tok/s | Decode story / code tok/s |
| --- | --- | --- |
| 5080 alone | 1,726-2,017 / 1,970 | 83-87 / 88-105 |
| 5080 + 3090, auto (K = 22) | 2,037 / 2,073 | 80 / 109 |
| 5080 + 3090, best (K = 26) | 2,039 / 2,357 | 84 / 110 |

- Prompts gain +18-20%. Decode is on par with the faster card alone.
- A third card (2080 Ti) made the pair slower: 68 / 90 tok/s.
- On one GPU the output is byte-identical to 0.1.20, and the hand-off is bit-exact (`--split-device 0`).
- On a split, conversation parking is refused, and `--resident-experts` falls back to `--mmap-experts`.

## 7. The prompt path

`src/prefill/prefill.cpp`, `include/strata/prefill/prefill.hpp`; GEMMs in `src/prefill/gemm.cu`,
`src/prefill/moe_mmq.cu`.

**Chunk size.**

- `--prefill N` (the engine's default is 0, the token path). `--prefill auto` (setup's) picks the largest chunk, up to
  8,192, whose buffers fit in the cache slots it borrows. A request borrows only what its prompt needs
  (`Prefill::relayout`). `--prefill auto:N` raises the cap (#282).
- **Borrowing.** By default the buffers are carved out of the **top (last) slots of the expert cache**. They are lent
  for the prompt and refilled after it. `--no-prefill-borrow` reserves the buffers for the whole session instead.
  Estimate without borrowing: 160 + chunk × 680/1024 MiB, which is ~1.5 GB at 2,048 (MULTI_GPU).
- A text part of ≤ `--short-read` (64) tokens goes through the verify windows instead ([STRATA_DECODE.md](STRATA_DECODE.md)).

**Per layer of a chunk.**

- The dense projections are tensor-core GEMMs (quantized weights dequantized to FP16; BF16 as is).
- The recurrences walk the chunk in one kernel.
- The routed experts are grouped by expert:
  - **resident experts** are read from their VRAM slots;
  - **the others stream from the host arena** through a ring on a copy stream.
- The expert products use **llama.cpp's quantized MMQ kernels**, in groups of 16 experts, for every layer whose types
  MMQ covers (always for the canonical Q2_0 pack). Otherwise there is a dequantize-to-FP16 path. MMQ replaced FP16
  expansion in 0.1.13. `STRATA_PREFILL_MMQ=0` is the A/B switch.
- The PLE block runs once for the whole chunk.

**The streaming rule.**

- From chunks of **`stream_all_min()` = 1,024 tokens** (`STRATA_PREFILL_STREAM_MIN`), nearly all 512 experts of a
  layer are routed. So **every non-resident expert of every layer streams in a fixed order** through a big ring:
  - 384 slots when ≥ 90% of the streamed bytes DMA straight from pinned RAM;
  - 96 when many go through host copies, or on a split (`STRATA_PREFILL_RING`).
- That lets the copy engine fetch the next layer's experts during this layer's attention half.
  - Measured on Q2_0 with 8,192-token chunks: 96 slots 1,153 tok/s, 384 slots 1,294.
  - IQ3_S on 64 GB with a third of the arena unpinned: 96 slots 1,216 tok/s, 256 slots 1,070.
- The floor dropped from 2,048 to 1,024 in 0.1.30. Measured on the 5070 (Q2_0 / IQ2_XS): 1,500-token prompts 621 to
  785 / 612 to 735 tok/s, 2,000-token 727 to 934 / 712 to 892, same output. Below ~1,000 tokens the output changed on
  Q2_0, so 1,024 is the floor.
- **Below 1,024**, only the routed misses are staged, one by one, through an 8-slot ring (`STAGE`).
- **Unpinned blobs** (the part of the arena the driver would not register) are copied into a pinned ring by stager
  threads ahead of the launches:
  - 2-4 threads with a 16-deep ring by default;
  - 32 threads with a 128-deep ring when the experts are read from the GGUF in place;
  - set with `STRATA_STAGER_THREADS` and `STRATA_STAGER_RING`.
  - Without stager threads, copying in line left the GPU idle for ~15 s of a 32K IQ3_S prompt.

**The next chunk's n-gram rows are read ahead.**

- In `Prefill::run`, the PLE rows of a whole chunk (16 per token) are one batched request to the n-gram reader
  (`gather_batch`), into one of two pinned host buffers.
- After chunk c's rows are uploaded, a `std::async` task starts gathering chunk c+1's rows into the other buffer. It
  starts once that buffer's previous upload has completed.
- Chunk c+1 then waits for that future instead of reading.

**Measured prompt speeds** (DETAILS, engine 0.1.26, tok/s, ±1 run per length):

| Model | 1K | 4K | 32K | 64K | 128K | 262K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q2_0 | 536 | 1,299 | 2,171 | 2,126 | 2,107 | 1,304 (0.1.22) |
| IQ3_XXS | 482 | 1,007 | 1,745 | 1,609 | 1,602 | - |
| **IQ3_S** | 427 | 913 | 1,624 | 1,640 | 1,443 | - |

Before 0.1.13 (2,048-token chunks, FP16 expansion), the paper measured 285-571 tok/s, flat with context. The paper
gives the cause (Finding 11): per-expert launches and FP16 dequantization, using ~7 GB/s of the 26 GB/s link. The
0.1.13 changes (chunk up to 8,192, MMQ, next-layer streaming, batched PLE, stager threads) took a 32K prompt from 572
to 1,290 tok/s on Q2_0, and from 383 to 1,208 on IQ3_S (`bench/results/2026-09-28-prefill-speed`).

## 8. Conversation checkpoints

`generate.cpp` around lines 825 and 4229; retention in `include/strata/program/conv_cache.hpp`; state copy in
`src/core/conversation_state.cpp`.

**What a checkpoint holds.** Only the *running* state, which cannot be rewound:

- the 36 GDN recurrences and conv histories;
- each QSA layer's indexer tail, spare key and block position;
- the PLE's normalized history;
- the token ids and image keys it covers.

The *positional* state (the QSA KV, pooled indexer keys, the drafter's KV) stays in place. Rewinding just rewrites
from that position. So a checkpoint is valid only while the session's cells below it still hold its tokens. The kept
checkpoints are always a chain of prefixes of the live tokens. **Size: ~118 MB of host RAM each** (DETAILS; the
`ConvCheckpoint` comment in `generate.cpp`). It is the same set the verifier snapshots to roll back drafts.

**Where one is taken** (`--serve` only):

- At the **last turn boundary**: the prompt is read up to the last `<|im_start|>` (`--turn-token`, 248045), a
  checkpoint is taken, and then the new turn's header is read.
- **Every `--prompt-cache-every` freshly read prompt tokens** (default **16,384**), at chunk ends. On a split, each
  stage saves its own part when it reaches that position.
- At the **end of the system prompt**, for a prompt read from token 0 whose first turn boundary is at
  ≥ `--prompt-cache-root` tokens (default **2,048**; PR #62 + #65). Smaller ones are cheaper to read again (~0.3 s).

**How many are kept, and what evicts them.**

- `--prompt-cache N` (default **6**; 0 = off).
- Over budget, the **root** (the shortest, in practice the system prompt) is **pinned**, and the rest leave by
  **least recent use**. A checkpoint's stamp advances on creation, on a re-save at its length, and on every resume
  through it.
- With fewer than 2 slots the pin is off and the oldest leaves.
- A request also drops any checkpoint that is not a prefix of its prompt, or that lies past its resume point.
- A changed control-vector state drops the cache once.

**Conversation parking** (opt-in; DETAILS, "Multiple conversations"):

- `--conversation-cache-mib` (default 0 = off) and `--conversation-cache-slots` (4) park whole conversations in host
  RAM: the running state, its checkpoints, the used KV pages and the drafter KV.
- When a conversation switches, the oldest entries are evicted first.
- Admission needs `--conversation-cache-min-free-mib` (2,560) of physical RAM free, checked before and after
  capture.
- KV pages that did not change are reused on the next park. Snapshots are not kept across restarts.
- Parking is refused with a layer split.

**Measured effect.** Strata's docs give no measured number. The design reasoning is that a 30K system prompt costs
~30 s to re-read at ~1,000 tok/s (`conv_cache.hpp`). The paper's "What could make it faster" estimates that a
30-second re-read becomes "a fraction of a second". Strata publishes no timing of the feature.

## 9. The n-gram reader

`src/ngram/ple_reader.cpp`, `include/strata/ngram/ple_reader.hpp`, `src/platform/direct_file.cpp`; options in
`generate.cpp` around line 2027.

**The table.** 320,001,536 rows of 90 B (IQ4_NL), 26.8 GiB (28.8 GB). Rows are 160 B in an FP8 variant. A token needs
16 rows (`PLE_N_HEADS`), on 16 different pages, all known as soon as the token id is known. The reader never holds the
table in RAM with `direct`.

**The read path.**

- `issue(rows, n, out)` runs as soon as the token is known:
  - serves row-cache hits by copying them;
  - turns the misses into **4 KiB-aligned page reads** (a row can span two pages);
  - **deduplicates pages within the ticket**;
  - **sorts by offset**, so a prompt chunk reads in near-sequential order;
  - appends them to the reader's queue.
- `collect(ticket)` blocks before layer 1 reads the rows.
- With `io_thread` (the default), one worker thread owns every file call. It submits queued jobs and reaps
  completions. `--ple-sync-submit` makes the caller's thread do it instead (A/B switch).
- **One FIFO for prompt and decode reads.** There is a single `queue` deque. `pump()` moves its head into the free
  in-flight slots, at most `--ple-inflight` deep, in arrival order. Tickets have no priority. A decode token's 16 rows
  issued behind a prompt chunk's thousands of page reads wait for them.
- Underneath, `DirectFile` opens the file with `O_DIRECT` (Linux) or `FILE_FLAG_NO_BUFFERING | OVERLAPPED`
  (Windows). Reads are issued by a **pool of threads**: `STRATA_IO_THREADS`, default **16** on Linux (each does a
  blocking `pread`, so the thread count is the queue depth) and **4** on Windows (overlapped submits). One issuing
  thread had capped prompt reads at ~80K reads/s, ~12 µs each.

**Switches and their defaults.**

| Switch | Default | What it does |
| --- | --- | --- |
| `--ple-io direct\|mmap\|ram` | `direct` | `direct`: unbuffered 4 KiB reads (above). `mmap`: the table mapped through the OS page cache (the paper's original design). `ram`: mapped and locked with `mlock`. Not on Windows |
| `--ple-inflight N` | 64 (1-1,024) | in-flight page reads per reader |
| `--ple-row-cache N` | 1,048,576 rows (~95 MB) | in-process row cache, set-associative with per-set replacement. 0 = off |
| `STRATA_IO_THREADS` | 16 Linux / 4 Windows (1-64) | issuing threads |
| `STRATA_SSD_KEEPALIVE` | 100 ms (0 = off) | when no read has gone out for this long, the worker reads one random page, so the SSD does not drop into a power state. Some SSDs stall the next reads 50-150 ms after ~250 ms idle (WD_BLACK SN7100, Windows 11). `direct` with the I/O thread only |
| `STRATA_SSD_KEEPALIVE_WINDOW` | 60 s | how long after the last `issue` the keep-alive continues |
| `--ple-delay-us N` | 0 | fault injection: every read completes no earlier than N µs after issue |
| `--ple-sync-submit` | off | submit on the caller's thread (A/B) |
| `--no-ple-prefetch` | off (prefetch on) | A/B for the `mmap` path's prefetch of a token's 16 rows. Measured without it: the gather took 2.10-2.61 ms/token, 16 separate page faults into the 26.8 GiB mapping |
| `--no-ple` | off | diagnostic ablation |

**Row cache hit rate.** Strata measured it offline, not in the engine, on its **"frozen corpus"**
(`bench/results/2026-09-23-ngram-io`, quoted in `ple_reader.hpp`). A 1M-row (~95 MB) cache would serve **up to ~82%**
of reads, and within one long prompt 20-34% of rows recur. That results directory is **not in the published
repository at this commit** (the oldest is `2026-09-24-final`), so the corpus cannot be inspected. The engine
reports its live hit count per run (`ReaderStats::cache_hits`), but no published record states it.

**Measured effect of the reader as a whole.** On the 2,048-chunk path the paper reported 16 row reads per token,
issued one step ahead and hidden behind the embedding and layer 0. UNSLOTH reports ~5 s per 4K chunk of PLE reads on
a slower setup. No decode-time breakdown of the reader is published for the default `direct` path. For this
machine's SSD, see [ADR-0015](../adr/0015-ngram-table-on-the-ssd.md) and the
[n-gram gather record](../measurements/MEASUREMENTS.md#2026-10-15-ngram-gather).

## 10. Sizes checked against Strata's docs

| Size | From the code | Strata's docs | Agree? |
| --- | --- | --- | --- |
| Arena, canonical Q2_0 | 33,973,862,400 B (24,576 × 1,382,400) | "34 GB" (paper Table 1; setup `arena_gb` 34.0) | Yes |
| Arena, native packs | per layer, from `native_experts.txt` | IQ2_XS 35.5 GB, IQ3_XXS 43 GB (paper); IQ3_S 50.3, Coder 23.4 (setup); "23-50 GB" (DETAILS) | Yes. The paper's Figure 1 says 43 GB for IQ3_XXS, setup says 42.9 |
| Cache slots | (free − reserve) / largest blob, or sized per blob for native packs | ~700 experts per GB; ~4,500 on 12 GB at 4K; 1,588-1,796 at 262K (paper Table 4) | Yes. ~700/GB holds for 1.38 MB blobs; IQ3_S's larger blobs give ~490/GB (*estimate*, section 2) |
| KV per token | int8: 1,056 B per QSA layer and cell; 13 layers in RAM when streamed | "~13.7 KB of RAM per context token, 1.7 GB at 128K" (DETAILS); "~13 KB" for scaled contexts | Yes (13 × 1,056 = 13,728 B). VRAM also holds 128 B per cell per layer of indexer keys and 256 B per cell of shared RoPE table (DETAILS: "~0.26 KB") |
| A checkpoint | GDN state + indexer tails + PLE history | "~118 MB each, up to 6" (DETAILS) | Yes. The code comment and the `--prompt-cache` help say the same |
| Hand-off per window | 51,216 B per token, ≤ 8 tokens | "a few hundred KB" (MULTI_GPU) | Yes |
| Prompt buffers on a split | 160 + chunk × 680/1024 MiB without borrowing | "1.5 GB each at the default 2048-token chunk" (MULTI_GPU) | Yes (1,520 MiB) |

Two places where the docs differ from the code:

- **8-bit KV starts above 8K, not 4K.** DETAILS' speed section says "8-bit KV above 4K". Setup writes `int8` only
  above 8,192, and the paper says "past 8K".
- **`--layer-split auto` uses a cost model.** MULTI_GPU describes "the most of the profile". The code minimises a
  predicted window time (section 6).
