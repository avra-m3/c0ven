# Measurements

The engine's measured numbers: what operations cost on an Intel Arc Pro B60 through Level Zero, what the engine's
kernels reach, and how fast the engine runs. Each row gives a value and the date it was measured. The date links
the row, and code and other pages link these rows for their numbers.

## The test machine

Every number here was measured on one machine:

- two Intel Arc Pro B60 cards (24 GB each), each on its own PCIe 5.0 x8 link to the CPU;
- an AMD Ryzen 9 7945HX: 16 Zen 4 cores on two CCDs, with AVX-512 (VNNI, VBMI, BF16, IFMA);
- 96 GiB of dual-channel DDR5 at 5200 MT/s, about 90 GiB of it visible to Linux;
- a PCIe 4.0 NVMe SSD, which holds the model files;
- Linux 6.17 with the in-tree `xe` driver, and the engine image's user-space stack: compute-runtime 26.35.39758.10,
  IGC 2.41.5 and the Level Zero loader 1.32.0 ([ADR-0002](adr/0002-pin-the-toolchain.md)).

Other software shares the machine. A number that rests on the host (the CPU's experts, host memory, load times)
moves with what else runs, so compare such numbers only within one row.

## How the numbers were taken

- A number is the median of at least five runs, or a range from least to most.
- "A → B" is a change judged against the code before it in ABBA order: rounds of A, B, B, A, at least three, so
  each arm runs six times. Each run is a fresh process with the same inputs
  ([ADR-0016](adr/0016-how-a-change-is-judged.md)). Both arms of one row ran in one sitting. One row's A is not the
  previous row's B, so gains from several rows do not add up.
- A kernel is timed after at least 50 ms of work on its card. An idle B60 runs near 1200 MHz, and a kernel timed
  there looks 1.7-2 times slower ([the clock](#the-clock)).
- A change gave the same tokens in both arms unless its row says otherwise.
- `doc-4k-qa`, `doc-32k` and the other named prompts are in `reference/prompts.jsonl`. `long-4k`, `long-32k`,
  `long-128k` and `long-256k` are longer prompts kept with the records, not in the repository, as are `s512`,
  `s1024` and `s4352`, named by their length in tokens.

## Level Zero on a B60

[LEVEL_ZERO.md](LEVEL_ZERO.md) gives the calls. This section gives what they cost.

### Where data moves

Kernel reads, kernel stores and copy-engine copies, with the rate each reaches.

```mermaid
flowchart LR
    SSD["NVMe SSD"]
    HOST["Host memory<br/>DDR5"]
    CPU["CPU<br/>16 cores"]
    C0["Card 0<br/>reads its VRAM at 455 GB/s"]
    C1["Card 1<br/>reads its VRAM at 455 GB/s"]

    SSD -->|"6.5 GB/s, O_DIRECT"| HOST
    HOST -->|"40-44 GB/s, ggml expert products"| CPU
    HOST -->|"28.3 GB/s, kernel reads"| C0
    HOST -->|"28.3 GB/s, kernel reads"| C1
    C0 <-->|"22 GB/s, kernel stores of 64 bytes or more"| C1
```

Both cards reading host memory at once reach about 45 GB/s together. A copy engine reads host memory at the
same 28.4 GB/s as a kernel. While a prompt runs, each card stages its experts from host memory at 19-22 GB/s.

### Memory

| Operation | Value | Measured |
| --- | --- | --- |
| Largest single device allocation (`zeMemAllocDevice`) | 22.71 GiB; 23.00 GiB with relaxed allocation limits | <a id="2026-10-11-device-memory"></a>[2026-10-11](#2026-10-11-device-memory) |
| Device memory one card holds | 25.31 GB. Past VRAM, allocations spill to system RAM and do not fail. Only Sysman's free-memory figure shows where VRAM ends. | [2026-10-11](#2026-10-11-device-memory) |
| Kernel reads from VRAM | 455 GB/s a card; 909.5 GB/s for both | [2026-10-11](#2026-10-11-device-memory) |
| Kernel reads from host memory (`zeMemAllocHost`, one context for both cards) | 28.3 GB/s a card; about 45 GB/s for both | <a id="2026-10-12-host-memory"></a>[2026-10-12](#2026-10-12-host-memory) |
| Host memory from `zeMemAllocHost` | The memory lock limit does not apply. Past 26-32 GiB the host swaps. Importing memory that already exists does not work. | [2026-10-12](#2026-10-12-host-memory) |
| Copy engine reading the process's own (anonymous) memory | 28.4 GB/s on both cards, the same as from `zeMemAllocHost` | <a id="2026-10-17-anon-arena-under-reclaim"></a>[2026-10-17](#2026-10-17-anon-arena-under-reclaim) |
| One copy from anonymous memory | about 1.3 µs more than the same copy from `zeMemAllocHost` | <a id="2026-10-17-anon-arena-default"></a>[2026-10-17](#2026-10-17-anon-arena-default) |
| Recording a copy list whose sources are anonymous memory | The runtime binds each copy's source when the list is recorded (21-46 µs a copy) and unbinds it when the list is freed (50-89 µs). | <a id="2026-10-17-swaps-keep-the-copies"></a>[2026-10-17](#2026-10-17-swaps-keep-the-copies) |
| Region copy (`zeCommandListAppendMemoryCopyRegion`) | half the rate of a plain copy | <a id="2026-10-17-staging-half-downs"></a>[2026-10-17](#2026-10-17-staging-half-downs) |
| Freeing 20 GiB of device memory, and mapping it again at the same addresses | 3.0-3.6 ms to free, 159-160 ms to map again | <a id="2026-10-16-vram-give-back"></a>[2026-10-16](#2026-10-16-vram-give-back) |
| Replaying a list recorded before such a free | Works up to 16 GiB a card. Past 16 GiB the driver crashes (SIGSEGV in `libze_intel_gpu`). | [2026-10-16](#2026-10-16-vram-give-back) |
| Freed VRAM in Sysman's free figure | at once, or within 0.1 s if the other card read it | [2026-10-16](#2026-10-16-vram-give-back) |
| Freeing and mapping host physical memory the same way | fails with `ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY` | [2026-10-16](#2026-10-16-vram-give-back) |
| VRAM held by an idle process with the runtime and every kernel loaded | 200 MiB a card | [2026-10-16](#2026-10-16-vram-give-back) |

### Launches and command lists

| Operation | Value | Measured |
| --- | --- | --- |
| 960 small kernels launched one by one | 1.28 ms (1.3 µs a launch) | <a id="2026-10-11-launch-cost"></a>[2026-10-11](#2026-10-11-launch-cost) |
| The same 960 kernels replayed as 48 recorded lists | 0.83 ms, of which 0.06 ms host time | [2026-10-11](#2026-10-11-launch-cost) |
| Values that change each token, given to recorded kernels | Pass them through memory. Mutable command lists and indirect launches also work. | [2026-10-11](#2026-10-11-launch-cost) |
| A kernel reading one word of host memory at its start | 0.8 µs | <a id="2026-10-16-bf16-products-occupancy"></a>[2026-10-16](#2026-10-16-bf16-products-occupancy) |
| Products that share inputs, launched together in a list with no ordering between them | a decode step 17.52 → 16.74 ms | <a id="2026-10-16-decode-launches-together"></a>[2026-10-16](#2026-10-16-decode-launches-together) |
| Two products fused into one launch, against the two launched together | 1.4-2.4 µs saved a pair | [2026-10-16](#2026-10-16-decode-launches-together) |
| Recording a copy list of about 35,000 copies | 45 ms; 8 ms of wait when recorded on a thread of its own | <a id="2026-10-16-swaps-and-prompts"></a>[2026-10-16](#2026-10-16-swaps-and-prompts) |
| Recording every list the engine uses, at load | 0.19 s | <a id="2026-10-16-load-phases"></a>[2026-10-16](#2026-10-16-load-phases) |

### Signals between the CPU and a card

| Operation | Value | Measured |
| --- | --- | --- |
| A spinning kernel seeing the CPU's write to host memory | Seen in 100 of 100 with a system-scope acquire fence before each volatile load. An atomic acquire load alone sees it in 21-37 of 100. | <a id="2026-10-11-doorbell"></a>[2026-10-11](#2026-10-11-doorbell) |
| Round trip CPU → card → CPU through a flag in host memory | 4.54 µs median, 5.88 µs p99 | [2026-10-11](#2026-10-11-doorbell) |
| The same round trip with events | about the same as the flag | [2026-10-11](#2026-10-11-doorbell) |

### Between the two cards

| Operation | Value | Measured |
| --- | --- | --- |
| Round trip of flags alone through host memory | 6.4 µs | <a id="2026-10-12-card-to-card"></a>[2026-10-12](#2026-10-12-card-to-card) |
| Exchange of partial sums through host memory, 1-4 tokens | 10.0-10.5 µs; 96 in a row 0.99-1.03 ms | [2026-10-12](#2026-10-12-card-to-card) |
| Hand-off of 205 KB through host memory | 50.3 µs | [2026-10-12](#2026-10-12-card-to-card) |
| One fp16 store per work-item into host memory | 13-54 times slower than wider stores, and the sums were wrong | [2026-10-12](#2026-10-12-card-to-card) |
| Exchange of 2,560 floats a row, an event per exchange, through host memory | 3.9-5.0 µs at one row; 10.2-10.3 µs at four | <a id="2026-10-16-peer-exchange"></a>[2026-10-16](#2026-10-16-peer-exchange) |
| The same exchange with the kernel writing into the other card's memory | 2.3-3.4 µs at one row; 5.1-5.2 µs at four | [2026-10-16](#2026-10-16-peer-exchange) |
| The same exchange as a copy into the other card | almost no saving at one row; 0.43 ms saved a decode step at four | [2026-10-16](#2026-10-16-peer-exchange) |
| A flag in the other card's memory in place of the event, 96 exchanges | 0.20 ms saved at one row, 0.16 ms at two; 0.05 ms lost at four | <a id="2026-10-16-peer-exchange-decode"></a>[2026-10-16](#2026-10-16-peer-exchange-decode) |
| Kernel stores into the other card's memory | 22.2-22.5 GB/s with stores of 64 bytes or more; 1.9 GB/s with one fp16 a work-item | <a id="2026-10-15-prompt-exchange-stores"></a>[2026-10-15](#2026-10-15-prompt-exchange-stores) |
| Sending a 4,352-row chunk's mixed activations to the other card | 0.61 ms with 128-byte block stores; 11.7 ms with 2-byte stores | <a id="2026-10-17-hc-split"></a>[2026-10-17](#2026-10-17-hc-split) |

### The clock

| Condition | Value | Measured |
| --- | --- | --- |
| Idle card | near 1200 MHz; a kernel timed there looks 1.7-2 times slower | <a id="2026-10-11-kernel-spike"></a>[2026-10-11](#2026-10-11-kernel-spike) |
| Work needed to leave idle | about 50 ms | [2026-10-11](#2026-10-11-kernel-spike) |
| During decode, read by Sysman every 1.17 ms | 2,400 MHz, the top of the range, in 99.74-100% of samples, with no throttle reason | <a id="2026-10-16-gpu-clock-in-decode"></a>[2026-10-16](#2026-10-16-gpu-clock-in-decode) |
| Waiting between decode plans | down to 2,250 MHz for at most about 2 ms | [2026-10-16](#2026-10-16-gpu-clock-in-decode) |

### What the driver reports

| Property | Value | Measured |
| --- | --- | --- |
| Extensions | every extension the design uses is listed | <a id="2026-10-11-level-zero-report"></a>[2026-10-11](#2026-10-11-level-zero-report) |
| Contexts | one context holds both cards | [2026-10-11](#2026-10-11-level-zero-report) |
| Copy engines | one per card, at queue group ordinal 1 | [2026-10-11](#2026-10-11-level-zero-report) |
| Atomics on host memory | `ATOMIC`, not `CONCURRENT_ATOMIC` | [2026-10-11](#2026-10-11-level-zero-report) |
| Matrix and dot-product units | DPAS and DP4A flags set | [2026-10-11](#2026-10-11-level-zero-report) |
| PCIe link speed through Sysman | not reported | [2026-10-11](#2026-10-11-level-zero-report) |

## Kernels on a B60

The engine's kernels are OpenCL C ([ADR-0003](adr/0003-kernel-language.md)). On the B60 the expert products read
257-344 GB/s at one token (56-76% of the card's 455 GB/s) and the dense products 370-421 GB/s. Experts in the
reordered layout run 3.2-8.8 times faster than in the file's layout
([2026-10-11](#2026-10-11-kernel-spike)).

### Decode kernels (1-4 rows)

"Step" is card 0's time in that kernel during one decode step, or the whole step where it says so.

| Kernel | Change | Kernel time | Step | Measured |
| --- | --- | --- | --- | --- |
| grouped expert products (`moe.cl`) | 32-bit offsets and vector loads; the kernels were bound by instructions, not memory | gate and up 2.0-2.4 times faster, down 1.2-2.1 times | 3.96 → 2.29 ms; 22.16 → 20.72 ms a token | <a id="2026-10-14-decode-expert-kernels"></a>[2026-10-14](#2026-10-14-decode-expert-kernels) |
| Q6_K, Q2_0 and i-quant decoders | bytewise subtraction in whole 32-bit words | Q6_K loop 187 instructions, was 299; Q6_K products about 4% faster | 17.34 → 17.255 ms a token, ranges overlapping | <a id="2026-10-15-sub-bytes-words"></a>[2026-10-15](#2026-10-15-sub-bytes-words) |
| products of up to 512 rows | K split over 4 sub-groups | 6.7-7.7 → 3.5-4.6 µs | five products 1.22 → 0.81 ms; 17.76 → 17.48 ms a token | <a id="2026-10-15-matvec-k-split"></a>[2026-10-15](#2026-10-15-matvec-k-split) |
| hyper-connections' down product | K split of 8 in place of 16 | 19.1 → 18.1 µs at one row | 18.07 → 18.03 ms a token; greedy agreement moved to 1,041 of 1,063 | <a id="2026-10-15-hc-down-split"></a>[2026-10-15](#2026-10-15-hc-down-split) |
| BF16 products | XMX units with bf16 activations | router's product 12.1 → 9.0 µs; PLE key product 169 → 123 µs | five products 1.41 → 1.11 ms; 19.13 → 18.82 ms a token; misses 25.01 → 25.12 a token from the rounding | <a id="2026-10-15-bf16-products-xmx"></a>[2026-10-15](#2026-10-15-bf16-products-xmx) |
| BF16 products | plan copied to each card in place of read from host memory | each launch 1.8-2.9 µs over a streaming read of its bytes; K split over work-groups 0.5-1.9 µs slower | 19.00 → 18.62 ms; decode 1.8-2.3% faster | [2026-10-16](#2026-10-16-bf16-products-occupancy) |
| out products (`ssm_out`, `attn_output`) | partial sums written in the card's memory and copied out, in place of written to host memory a float at a time | Q6_K at two rows 24.8 → 21.4 µs, four rows 35.6 → 26.3 µs; one row unchanged (19.9 against 20.0) | decode at long-4k 69.79 → 71.06 tokens/s | <a id="2026-10-16-out-products-host-writes"></a>[2026-10-16](#2026-10-16-out-products-host-writes) |
| `router_group` | router and expert grouping in one launch | 0.822 → 0.688 ms | 19.17 → 19.07 ms a token, ranges overlapping | <a id="2026-10-15-router-group"></a>[2026-10-15](#2026-10-15-router-group) |
| `router_group` | threshold top 10 and the job written by a work split | one row 14.6 → 5.3 µs | 0.685 → 0.421 ms; 17.88 → 17.67 ms a token | <a id="2026-10-15-router-top10"></a>[2026-10-15](#2026-10-15-router-top10) |
| `topk_select_few` (was `topk_rows`) | rank only a few candidates; same result, ties included | one row of 124,160 logits 281.3 → 26.2 µs | 20.69 → 20.19 ms a token | <a id="2026-10-14-topk-rows"></a>[2026-10-14](#2026-10-14-topk-rows) |
| `idx_cand`, `idx_merge` | no host-memory stores and no serial search | 37.0 → 9.8 µs and 57.1 → 15.7 µs | 20.32 → 19.56 ms a token; prompt 2,069 → 2,088 tokens/s | <a id="2026-10-15-indexer-selection"></a>[2026-10-15](#2026-10-15-indexer-selection) |
| `idx_cand` top-513 selection | faster selection; merging hot bins alone was not enough | one launch at 30,834 values 80-279 → 59-155 µs | 1.865 → 0.92 ms a step at 246K positions; long-128k decode 63.48 → 64.58 tokens/s | <a id="2026-10-16-indexer-select"></a>[2026-10-16](#2026-10-16-indexer-select) |
| `gdn_conv`, `gdn_recur` | DeltaNet norm and gates moved in; two launches a layer fewer; same bits | 0.98 → 0.78 ms | 18.04 → 17.85 ms a token | <a id="2026-10-15-gdn-fusion"></a>[2026-10-15](#2026-10-15-gdn-fusion) |
| kernels that make a8 activations | write them directly; `quantize_a8` and five launches a layer gone | 0.69 ms a card removed, 0.15 ms added elsewhere | 19.35 → 19.11 ms a token, ranges overlapping | <a id="2026-10-15-a8-fusion"></a>[2026-10-15](#2026-10-15-a8-fusion) |

### Prompt kernels (chunks of up to 4,352 rows)

"Chunk" is each card's time in that kernel for one chunk.

| Kernel | Change | Kernel time | Chunk and prompt rate | Measured |
| --- | --- | --- | --- | --- |
| prompt router | one sub-group per token | 4,159 tokens 987 → 29.3 µs | about 48 → 1.54 ms; 2,087 → 2,135 tokens/s | <a id="2026-10-15-prompt-router"></a>[2026-10-15](#2026-10-15-prompt-router) |
| `p_moe_scatter` | scatter each token's row once, in place of gathering the experts' activations | 2.74 → 2.03 ms | 131.8 → 98.5 ms; 2,135.5 → 2,174.5 tokens/s | <a id="2026-10-15-prompt-moe-scatter"></a>[2026-10-15](#2026-10-15-prompt-moe-scatter) |
| `p_moe_scatter` | sub-group block reads and writes; it was bound by its stores | 2.02 → 0.77-0.79 ms (271-276 GB/s) | 98.4 → 33.6 ms; 2,594 → 2,698.5 tokens/s | <a id="2026-10-15-prompt-moe-scatter-blocks"></a>[2026-10-15](#2026-10-15-prompt-moe-scatter-blocks) |
| `dequant_f16` | packed words and whole-line stores | 246-501 → 50-113 µs | 46.9 → 10.45 ms; 2,173 → 2,212.5 tokens/s | <a id="2026-10-15-prompt-dequant-stores"></a>[2026-10-15](#2026-10-15-prompt-dequant-stores) |
| bf16 products | tiles of 4 tokens where K or N is short; same bits | down 1,129 → 884 µs; up 955 → 783-803 µs; ab 77-82 → 48.5 µs | 2,261.5 → 2,351.5 tokens/s | <a id="2026-10-15-prompt-gemm-tiles"></a>[2026-10-15](#2026-10-15-prompt-gemm-tiles) |
| products with weights larger than L2 | bands of rows; same bits; no gain below L2 | qkv 2.20 → 1.86 ms; q 2.62 → 2.40 ms; PLE keys 7.66 → 3.82 ms | 2,417 → 2,505 tokens/s | <a id="2026-10-15-prompt-gemm-bands"></a>[2026-10-15](#2026-10-15-prompt-gemm-bands) |
| hyper-connections' low-rank down product (`gemm.cl`) | tiles of 32 tokens by 32 rows, 11 sub-groups a work-group; same bits; slower at 256 rows and fewer | 4,352 rows 0.911-0.917 → 0.775-0.782 ms; 1,024 rows 0.293-0.351 → 0.180-0.222 ms | long-128k first token 0.5% sooner | <a id="2026-10-17-hc-lowrank"></a>[2026-10-17](#2026-10-17-hc-lowrank) |
| expert gate and up | load each block's shared fields once (`block_raw`); bound by the decoding's loads | IQ2_S 7.27 → 5.66 ms; IQ3_S 7.15 → 5.74 ms | 329.4 → 284.1 ms; 2,216 → 2,275.5 tokens/s | <a id="2026-10-15-prompt-expert-block-fields"></a>[2026-10-15](#2026-10-15-prompt-expert-block-fields) |
| expert down | early loads, block writes and half tiles together | IQ4_NL 3.59 → 2.84 ms; Q2_0 2.97 → 2.62 ms | 178.7 → 148.9-149.1 ms; 2,545.5 → 2,599 tokens/s | <a id="2026-10-15-prompt-expert-down"></a>[2026-10-15](#2026-10-15-prompt-expert-down) |
| tiled attention | mask computed once a tile; the mask was about a quarter of the kernel | 4,159 rows 9.02-9.03 → 7.16 ms | 106.5-106.8 → 84.5 ms; 2,509 → 2,545 tokens/s | <a id="2026-10-15-prompt-attention-mask"></a>[2026-10-15](#2026-10-15-prompt-attention-mask) |
| Gated DeltaNet recurrence | one loop and one barrier a token; faster forms that change the sums lost greedy agreement | 2.60 → 2.46 ms | 93.6-94.3 → 88.8-89.3 ms; 2,685 → 2,708.5 tokens/s | <a id="2026-10-15-prompt-gdn-fused-loop"></a>[2026-10-15](#2026-10-15-prompt-gdn-fused-loop) |
| `p_idx_score` | read each pooled key once for eight rows; every score bit-identical | 4,352 rows at position 239,360: 49.0-52.7 → 20.7 ms a layer; at 20,000: 7.1 → 6.1 ms | long-128k first token 2.5-2.9% sooner | <a id="2026-10-17-idx-score-rows"></a>[2026-10-17](#2026-10-17-idx-score-rows) |
| `p_hc_combine_norm` | write partial sums into the other card directly, in 64-byte stores | 243-246 → 136-139 ms | 2,353.5 → 2,427 tokens/s | [2026-10-15](#2026-10-15-prompt-exchange-stores) |
| hyper-connections | each card computes half the rows and sends the other card its half; only from 2,048 rows | 4,352 rows: 0.85 → 0.43 ms, plus 0.61 ms to send | s4352 first token 1,663 → 1,559 ms; long-128k 53,800 → 51,110 ms | [2026-10-17](#2026-10-17-hc-split) |

## The host

| Operation | Value | Measured |
| --- | --- | --- |
| ggml's CPU backend on i-quant expert products | 5.0-5.7 GB/s a core; 40-44 GB/s on 16 cores; bound by arithmetic | <a id="2026-10-11-cpu-bar"></a>[2026-10-11](#2026-10-11-cpu-bar) |
| ggml on IQ4_XS and IQ4_NL | about 50 GB/s on 8 cores; bound by memory | [2026-10-11](#2026-10-11-cpu-bar) |
| ggml on Q2_0 | scalar code, 2.3 GB/s a core | [2026-10-11](#2026-10-11-cpu-bar) |
| ggml on 2-4 tokens at once | no reuse across the tokens | [2026-10-11](#2026-10-11-cpu-bar) |
| CPU's answer for one missed layer | 20-25 µs, plus 60-62 µs for each distinct expert, plus 19-25 µs for each further row on an expert already in the job | <a id="2026-10-16-cpu-answer-per-layer"></a>[2026-10-16](#2026-10-16-cpu-answer-per-layer) |
| Card 0 waiting for the CPU, decode on long-4k | 2.89, 4.31, 6.09 and 8.11 ms a step at 1, 2, 3 and 4 rows (16-26% of the step) | [2026-10-16](#2026-10-16-cpu-answer-per-layer) |
| One expert on the CPU, `compute_many` in two pool jobs | 88-93 → 80-85 µs; 18.34 → 18.15 ms a token. Workers polling for 5 ms between jobs made the mean and tail steps worse, so workers sleep. | <a id="2026-10-15-cpu-miss-path"></a>[2026-10-15](#2026-10-15-cpu-miss-path) |
| One expert on the CPU, `compute_many` in one pool job | 80.9-81.3 → 73.7-75.0 µs; 18.23 → 18.08 ms a token, ranges overlapping; same bits | <a id="2026-10-15-cpu-one-job"></a>[2026-10-15](#2026-10-15-cpu-one-job) |
| An expert's first miss on arena pages not yet touched | 268 µs with about 60 major faults, against 88 µs. Reading one byte a page at load removes the faults: 19.12 → 18.44 ms a token. | <a id="2026-10-15-arena-touch"></a>[2026-10-15](#2026-10-15-arena-touch) |
| Decode steps while the host is short of memory | 0.3-9 s, from about 60 major faults an expert. Giving back the model file's page cache removes most. | <a id="2026-10-14-decode-stalls"></a>[2026-10-14](#2026-10-14-decode-stalls) |
| `xe` moving the engine's host buffers (12.2 GB) from GTT to system placement under reclaim | decode stalls of 6.92 s and 2.75 s, then 566,270 major faults over the next 5,415 steps | <a id="2026-10-17-decode-stalls-eviction"></a>[2026-10-17](#2026-10-17-decode-stalls-eviction) |
| The same reclaim (one 54.8 GB page-cache read) with the arena in anonymous memory | worst step 48.1 ms, against a 5.75 s stall with the driver's arena; same tokens | [2026-10-17](#2026-10-17-anon-arena-under-reclaim) |
| Pinned host memory | given in contiguous pages; about 12 GiB available clean on 2026-10-13 | <a id="2026-10-13-design-inputs"></a>[2026-10-13](#2026-10-13-design-inputs) |
| Reading the model file with `O_DIRECT` | 6.49-6.64 GB/s with 8 threads of 1 MiB, on a quiet host | <a id="2026-10-16-load-read-rates"></a>[2026-10-16](#2026-10-16-load-read-rates) |
| Rearranging experts into the cards' layout (`to_runs`, 16 threads) | 12.26-13.04 GB/s | [2026-10-16](#2026-10-16-load-read-rates) |
| Gathering doc-4k-qa's 66,544 n-gram rows with an empty page cache | 37.6 ms; 256 threads 31.2 ms against 33.6 ms with 128; Linux AIO 76.9-84.8 ms; `io_uring` is refused in the container | <a id="2026-10-15-ngram-gather"></a>[2026-10-15](#2026-10-15-ngram-gather) |

## The engine

### Current figures

| Figure | Value | Measured |
| --- | --- | --- |
| Decode of doc-4k-qa | 17.255 ms a token | [2026-10-15](#2026-10-15-sub-bytes-words) |
| Prompt of doc-4k-qa (4,159 tokens) | 2,135 tokens/s | [2026-10-15](#2026-10-15-prompt-router) |
| `greedy_test` against llama.cpp, prompts of up to 512 tokens | 1,046 of 1,063 positions agree | [2026-10-15](#2026-10-15-matvec-k-split) |

### Start, load, sleep and wake

| Operation | Value | Measured |
| --- | --- | --- |
| Cold load at a 262,144-token context | 63.2-71.5 s (median 64.4), 55.4-63.0 s of it the experts | [2026-10-16](#2026-10-16-load-phases) |
| Copying the 20.49 GiB arena out of the mapped file on one thread | 39.4-46.2 s | [2026-10-16](#2026-10-16-load-phases) |
| Warm-up chunk at load | 1.9 s | [2026-10-16](#2026-10-16-load-phases) |
| Draft layer at load | 42.8 s more, nearly all quantising on the CPU | [2026-10-16](#2026-10-16-load-phases) |
| Host memory swapped out by a load | 2.4 GiB; 11-53 MB with pages given back per layer | [2026-10-16](#2026-10-16-load-phases) |
| Engine child to READY | 86.0-94.2 s, 8.7-8.9 s of it Python reading the GGUF twice | <a id="2026-10-16-engine-start-and-quit"></a>[2026-10-16](#2026-10-16-engine-start-and-quit) |
| Front end alone, to serving | 5.3-5.6 s, 4.4 s of it the same GGUF read | [2026-10-16](#2026-10-16-engine-start-and-quit) |
| QUIT | VRAM back 0.44-0.64 s after the line; the child exits after 0.61-1.11 s | [2026-10-16](#2026-10-16-engine-start-and-quit) |
| Host memory pinned at the defaults (8 checkpoints) | about 20.5 GiB of arena plus about 4 GiB a checkpoint | [2026-10-16](#2026-10-16-engine-start-and-quit) |
| Start to READY, weights read with `O_DIRECT` | 24.5-64.7 s; 19.9-54.3 s of it the load, which reads 58.95 GB | <a id="2026-10-17-sleep-and-wake"></a>[2026-10-17](#2026-10-17-sleep-and-wake) |
| Sleep, arena kept | 376-718 ms (median 393) | [2026-10-17](#2026-10-17-sleep-and-wake) |
| Wake | 14.2-20.7 s (median 15.0), 11.9-18.1 s of it reading the cards' 38.30 GB; same greedy tokens as before the sleep | [2026-10-17](#2026-10-17-sleep-and-wake) |
| VRAM held while asleep | 0.31-0.32 GiB a card | [2026-10-17](#2026-10-17-sleep-and-wake) |
| Page cache growth during a load | at most 0.79 GiB | [2026-10-17](#2026-10-17-sleep-and-wake) |
| Host memory swapped out | 0-50 MiB by a wake; 2.7-790 MiB by a start, most when the host stalled on compaction | [2026-10-17](#2026-10-17-sleep-and-wake) |

### Prompts

| Operation | Value | Measured |
| --- | --- | --- |
| Chunk of 64-1,024 tokens | 650-765 ms, of which 166-484 ms is the cards waiting for the 14.45 GB of uncached experts each card stages at 19-22 GB/s | <a id="2026-10-17-prefill-timings"></a>[2026-10-17](#2026-10-17-prefill-timings) |
| Chunk of 4,352 tokens | 1,645-1,680 ms (2,590-2,646 tokens/s) at position 0; 2,748 ms at position 239,360 | [2026-10-17](#2026-10-17-prefill-timings) |
| One attention layer in a 4,352-token chunk | 41 ms at position 0; 132 ms at position 239,360 | [2026-10-17](#2026-10-17-prefill-timings) |
| Prompt of 246,674 tokens | 136.0 s (1,813.6 tokens/s) | [2026-10-17](#2026-10-17-prefill-timings) |
| Saving or restoring a checkpoint | 15-180 ms | [2026-10-17](#2026-10-17-prefill-timings) |
| Prompts of up to 64 tokens read as decode windows | first token: 1 token 465.9 → 19.7 ms; 16 tokens 466.4 → 77.8 ms; 64 tokens 483.5 → 330.5 ms; 65 tokens and more unchanged | <a id="2026-10-17-prompt-windows"></a>[2026-10-17](#2026-10-17-prompt-windows) |
| One token after a checkpoint restore, as a decode window | 463.8-504.2 → 17.0-17.6 ms at 4K positions; 21.9-23.2 ms at 32K and 128K | [2026-10-17](#2026-10-17-prompt-windows) |
| Each card staging only its half of every uncached down product | staged bytes 14.45 → 10.07 GB a chunk; copy list busy time 656 → 481 ms at 64 rows; first token 25-32% sooner for 64-1,024 tokens, 3-5% at 4,018-31,338, 0.7% (inside the spread) at 112,981 | [2026-10-17](#2026-10-17-staging-half-downs) |
| Copy list recorded once, swaps copying into the outgoing expert's slot | s512 first token 815 → 488 ms; s4352 2,418 → 1,549 ms; long-4k 2,565 → 1,472 ms | [2026-10-17](#2026-10-17-swaps-keep-the-copies) |
| Arena in anonymous memory with region copies (`copy_rows`), against the driver's arena | 64-token prompt 294.2 → 282.8 ms; 1,024 and 4,352 tokens, decode and load unchanged | [2026-10-17](#2026-10-17-anon-arena-default) |
| Next chunk's n-gram rows read while this chunk runs | first token 1.3-1.5% sooner for long-32k and long-128k; one-chunk prompts unchanged | <a id="2026-10-17-ngram-ahead"></a>[2026-10-17](#2026-10-17-ngram-ahead) |
| Swap rounds held while a prompt is read | long-32k 2,088.8 → 2,132.2 tokens/s; long-4k unchanged | [2026-10-16](#2026-10-16-swaps-and-prompts) |

### Decode

| Operation | Value | Measured |
| --- | --- | --- |
| Both cards on every layer, the CPU taking the misses | 9.8-11.3 ms a token; one card alone 14.9-15.6 ms | [2026-10-13](#2026-10-13-design-inputs) |
| Drafting with the MTP layer and adaptive swaps | long-4k 54.76 → 65.31 tokens/s; long-32k 50.31 → 66.99; prompt 1.6-2.1% slower | <a id="2026-10-16-drafting-and-swaps"></a>[2026-10-16](#2026-10-16-drafting-and-swaps) |
| One verify-window row | about 5 ms (about 70 more CPU experts) | <a id="2026-10-16-draft-rounds"></a>[2026-10-16](#2026-10-16-draft-rounds) |
| Drafts sampled as the request samples, chain drafts at the catch-up's cell, q and k norms + 1 | long-4k 54.33 → 58.45 tokens/s; 1.579 → 1.645 tokens a step | [2026-10-16](#2026-10-16-draft-rounds) |
| A round's commit and catch-up; one chain draft | 2.42 → 1.97 ms; 1.39 → 1.34 ms; same drafts | <a id="2026-10-16-draft-layer-passes"></a>[2026-10-16](#2026-10-16-draft-layer-passes) |
| Experts cached per layer at a 262,144-token context | 288 → 324 without the draft layer, 267 → 303 with it; long-4k decode 52.55 → 54.76 tokens/s, prompt 2,474.6 → 2,555.4 tokens/s | <a id="2026-10-16-more-experts-cached"></a>[2026-10-16](#2026-10-16-more-experts-cached) |
| Card 0 waiting for the host, a decode plan at long-4k | 0.73 ms | <a id="2026-10-16-host-glue"></a>[2026-10-16](#2026-10-16-host-glue) |
| A step's n-gram rows read at once | layer 0's wait 0.91 → 0.19 ms a plan at long-4k, 2.18 → 0.25 ms at long-256k; long-32k decode 70.32 → 74.49 tokens/s | [2026-10-16](#2026-10-16-host-glue) |
| Exchanges written into the card's memory and copied into the other card's, from two rows | decode 0.9-2.0% faster: long-4k 72.19 → 72.88 tokens/s, long-256k 69.55 → 70.98 ([ADR-0030](adr/0030-exchanges-into-the-other-card.md)) | [2026-10-16](#2026-10-16-peer-exchange-decode) |
| KV cache read through a page table kept by Python ([ADR-0033](adr/0033-paged-kv-kept-by-python.md)) | no measurable cost: doc-32k 18.48 → 18.62 ms a step; the table's reads add about 11 µs to a step | <a id="2026-10-17-paged-kv"></a>[2026-10-17](#2026-10-17-paged-kv) |
| Arena in anonymous memory, against the driver's arena | 16.58 and 16.56 → 16.54 and 16.61 ms a step; the driver's host buffers 9.07 → 0.06 GB | <a id="2026-10-17-anon-arena"></a>[2026-10-17](#2026-10-17-anon-arena) |

### Against llama.cpp

One request at a time through each server's `/v1/chat/completions`, greedy, 512 tokens out, one slot at a
262,144-token context, the same IQ3_S file. C0ven runs with its defaults; llama.cpp is release b11719 with its SYCL
backend, its own placement (`--fit`, which keeps the experts of 18 of the 49 layers in host memory) and ubatch 1,024.
Prompt reading is the prompt's tokens over the time to the first token, decode the tokens after the first over the
time to the last, both at the client. Six fresh server processes each; C0ven's value is given first.

| Test | Value | Measured |
| --- | --- | --- |
| Prompt reading, doc-4k-qa (4,159 tokens) | 2,018.6 (2,011.5-2,021.9) against 356.9 (278.8-358.2) tokens/s; llama.cpp's lowest is its one run with the model file out of the page cache | <a id="2026-10-18-against-llama-cpp"></a>[2026-10-18](#2026-10-18-against-llama-cpp) |
| Prompt reading, doc-32k (32,010 tokens) | 2,066.2 (2,061.4-2,067.4) against 353.9 (347.1-354.1) tokens/s | [2026-10-18](#2026-10-18-against-llama-cpp) |
| Decode after doc-4k-qa | 85.62 (68.15-86.36) against 16.97 (16.92-17.06) tokens/s | [2026-10-18](#2026-10-18-against-llama-cpp) |
| Decode after doc-32k | 75.03 (74.58-75.37) against 14.35 (14.27-14.43) tokens/s | [2026-10-18](#2026-10-18-against-llama-cpp) |
| Time to the first token, doc-4k-qa | 2.060 (2.057-2.068) against 11.653 (11.611-14.917) s | [2026-10-18](#2026-10-18-against-llama-cpp) |
| Time to the first token, doc-32k | 15.492 (15.483-15.528) against 90.454 (90.402-92.231) s | [2026-10-18](#2026-10-18-against-llama-cpp) |
| Greedy texts in six processes | C0ven one for each prompt; llama.cpp five for doc-4k-qa and two for doc-32k, four of whose six equal C0ven's | [2026-10-18](#2026-10-18-against-llama-cpp) |

### One card

The same tests with `ZE_AFFINITY_MASK=0`: the engine on card 0 alone, with 78 experts a layer in VRAM against two
cards' 307, and a 39.7 GiB arena. One card and two cards ran in one ABBA series, six fresh server processes each, with
the cards' VMs without page faults (`NEOReadDebugKeys=1 NEO_EnableRecoverablePageFaults=0` in the environment).
One card's value is given first.

| Test | Value | Measured |
| --- | --- | --- |
| Prompt reading, doc-4k-qa (4,159 tokens) | 930.9 (876.4-940.2) against 2,004.3 (1,850.9-2,016.9) tokens/s | <a id="2026-10-19-one-card"></a>[2026-10-19](#2026-10-19-one-card) |
| Prompt reading, doc-32k (32,010 tokens) | 1,122.5 (1,121.0-1,123.9) against 2,064.9 (2,057.0-2,069.5) tokens/s | [2026-10-19](#2026-10-19-one-card) |
| Decode after doc-4k-qa | 36.17 (33.24-39.40) against 84.41 (46.87-85.83) tokens/s | [2026-10-19](#2026-10-19-one-card) |
| Decode after doc-32k | 31.15 (29.40-32.70) against 73.16 (67.49-74.81) tokens/s | [2026-10-19](#2026-10-19-one-card) |
| Time to the first token, doc-4k-qa | 4.468 (4.423-4.745) against 2.075 (2.062-2.247) s | [2026-10-19](#2026-10-19-one-card) |
| Time to the first token, doc-32k | 28.516 (28.480-28.555) against 15.502 (15.468-15.561) s | [2026-10-19](#2026-10-19-one-card) |
| Greedy texts in six processes | one for each prompt on each arm. One card's doc-32k text differs from two cards' and llama.cpp's reference from its first token, and ends at 462 tokens | [2026-10-19](#2026-10-19-one-card) |
| Host memory for a one-card load | the process's anonymous memory peaked at 42.5-43.3 GiB | [2026-10-19](#2026-10-19-one-card) |

### Tried and not adopted

| Idea | Value | Measured |
| --- | --- | --- |
| Copying likely misses to a card ahead of time | 21.83 → 23.88 ms a token, slower. The next layer's router finds 54% of misses with 10 guesses; the copies slow the cards' kernels 5-12%. | <a id="2026-10-14-expert-prefetch"></a>[2026-10-14](#2026-10-14-expert-prefetch) |
| KV streaming, 32,768 resident cells per attention layer ([ADR-0008](adr/0008-kv-cache.md) item 3) | deterministic decode 1.5-3.9% faster from 32-40 more cached experts a layer, lost with drafting (77.32 → 73.97 tokens/s); reading the KV from host memory in place costs 50% of a decode step | <a id="2026-10-17-kv-streaming"></a>[2026-10-17](#2026-10-17-kv-streaming) |
| Other draft rules, 270 replayed on long-256k | best 64.34 tokens/s against the engine's 64.01 (up to 3 drafts, each at probability 0.5 or more) | <a id="2026-10-16-draft-rule-simulated"></a>[2026-10-16](#2026-10-16-draft-rule-simulated) |
| Draft layer on `mv_xmx_bf16` and `hc_up_mix_xmx`, plan on the card | 0.15-0.45 ms a pass faster; long-32k 73.54 → 76.67 tokens/s, but long-4k 70.69 → 69.92, and its rounding moved | <a id="2026-10-16-draft-layer-xmx"></a>[2026-10-16](#2026-10-16-draft-layer-xmx) |
| A smaller draft head | chain draft 0.98 ms; first drafts accepted 0.798 → 0.749 | [2026-10-16](#2026-10-16-draft-layer-passes) |

### Routing and requests

| Operation | Value | Measured |
| --- | --- | --- |
| Routed experts served by the cards, 24 a layer of 17,581 | 89.6% from a static cache, hottest first; 95.9% with swaps | <a id="2026-10-12-expert-hit-rates"></a>[2026-10-12](#2026-10-12-expert-hit-rates) |
| Layers with no miss at 3 tokens | 26.9%; 46.1% with swaps | [2026-10-12](#2026-10-12-expert-hit-rates) |
| Four requests sent within 0.1 ms | Without a wait the engine planned before reading all four in 10 of 51 runs, and the late ones reached their first token 0.55-1.78 s later. With up to 2 ms of wait after a GEN to an idle engine, 0 of 24. | <a id="2026-10-17-requests-sent-together"></a>[2026-10-17](#2026-10-17-requests-sent-together) |

### Agreement with references

| Comparison | Value | Measured |
| --- | --- | --- |
| Image encoder against an F32 computation, 4,096 tokens | mean cosine 0.9998-0.99995, least 0.960-0.998 | <a id="2026-10-17-encoder-reference"></a>[2026-10-17](#2026-10-17-encoder-reference) |
| llama.cpp's image encoder against an F32 computation | least cosine 0.990 without flash attention; mean 0.957 with it | [2026-10-17](#2026-10-17-encoder-reference) |
| Image encoder time, 4,096 tokens on two cards | 1.75-1.76 s | [2026-10-17](#2026-10-17-encoder-reference) |
| Video encoder against llama.cpp, two frame pairs | mean cosine 0.99991-0.99992, least 0.9962-0.9983 | <a id="2026-10-17-video-pair-reference"></a>[2026-10-17](#2026-10-17-video-pair-reference) |
| llama.cpp's CPU video encoder against an F64 computation | least cosine 0.9991 at 64 tokens, 0.9966 at 252; 0.34 with the frames swapped | [2026-10-17](#2026-10-17-video-pair-reference) |
| Video encoder time, one frame pair | 38.5-40.9 ms | [2026-10-17](#2026-10-17-video-pair-reference) |

### The engine image

| Check | Value | Measured |
| --- | --- | --- |
| The root Dockerfile | builds with nothing copied to the host; every version matches [ADR-0002](adr/0002-pin-the-toolchain.md) | <a id="2026-10-13-engine-image"></a>[2026-10-13](#2026-10-13-engine-image) |
| A container as user `ubuntu` | sees both B60s, compiles for `bmg-g21`, has an unlimited memory lock and reads `/models` | [2026-10-13](#2026-10-13-engine-image) |
