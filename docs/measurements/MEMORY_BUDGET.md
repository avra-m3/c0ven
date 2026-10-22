# Memory budget

What the IQ3_S file and the engine's state take on each B60 and in RAM, from the real tensor sizes.

Every row is labelled **computed** (summed from sizes in the repository, with the arithmetic or source next to it) or
*estimate* (a runtime overhead or a free-RAM figure not yet measured, with its method).

## The answer

This budget is for the tensor split ([ADR-0023](../adr/0023-two-cards-tensor-split.md)), with the free RAM measured.

Each card holds half of everything: half the dense weights, the embedding, the head, the PLE and the draft layer,
half of each sequence's state, and half of every cached expert. That leaves 19.9 GB per card before the
per-sequence reserve. It caches 18,968 experts (77.2%) for one sequence at 262K, and 17,489 (71.2%) for four
([tables](#per-card-for-the-tensor-split)).

An arena of every expert does not fit in the host memory that can be pinned. Level Zero host memory comes from large
contiguous pages, and free memory alone does not decide how much the host gives. Measured on 2026-10-13 with
nothing else on the cards ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)): `MemAvailable` was
51.5-55.2 GiB, but the host's free 2 MiB blocks ran out after ~12 GiB. After that every GiB made the kernel compact
memory and reclaim the page cache, and past ~25 GiB the host began to swap. On 2026-10-12 the same probe
reached 45 GiB before swapping. So the limit depends more on how fragmented the host's memory is than on how much is
free. The arena of every expert is 46.84 GiB, and 50.06 GiB with one 262K sequence's streamed KV.

An arena of only the experts the cards do not hold fits: 10.3-10.7 GiB for one sequence, 13.9 GiB pinned with one
262K sequence's streamed KV, 26.4 GiB with four ([RAM](#ram-for-the-tensor-split)). A swap then moves an expert both
ways on the copy engines: the one leaving VRAM back to the arena, the one entering from it. The engine uses this
arena ([ADR-0023](../adr/0023-two-cards-tensor-split.md) item 5).

The core does not take this plan yet. It fits its own expert count, keeps every slot's whole-context KV in VRAM, and
its checkpoints hold that KV too ([ADR-0027](../adr/0027-the-c-abi-as-the-core-was-optimised.md) items 2, 3 and 6). Its
arena measures 20.49 GiB at 288 experts a layer ([record](MEASUREMENTS.md#2026-10-16-load-phases)) and 17.20 GiB at
324 ([record](MEASUREMENTS.md#2026-10-16-more-experts-cached)).
With swaps on, the pages no request holds lend their memory to the expert cache
([ADR-0045](../adr/0045-the-kv-pool-and-the-expert-cache-share-each-cards-memory.md)). At one 262,144-token slot with the draft layer it caches 307 experts a layer; with a pool of 32,768 cells and
every indexer key in VRAM it would cache 339, which makes decode 1.5-3.9% faster without drafting
([record](MEASUREMENTS.md#2026-10-17-kv-streaming)).

The n-gram table stays on the SSD ([ADR-0015](../adr/0015-ngram-table-on-the-ssd.md)).

## How the numbers are computed

A script, not in the repository, does all the sums. It:

1. Reads [model/tensors-IQ3_S.tsv](../model/tensors-IQ3_S.tsv) (1,224 rows) and groups each `blk.N.*` row by layer `N`
   and by its `category` column (`experts`, `ple`, otherwise dense). It checks that the experts sum to
   50,292,326,400 B and the dense tensors to 3,576,097,280 B, the totals of
   [the tensor list](../model/tensors-IQ3_S.tsv).
2. Marks layer `N` an attention layer if the file has `blk.N.attn_k.weight`, and a DeltaNet (GDN) layer if it has
   `blk.N.ssm_a`. That gives attention layers 3, 7, 11, ..., 47 (12) and 36 GDN layers.
3. Takes per-token and per-sequence sizes from [MODEL_LAYER.md](../spec/MODEL_LAYER.md) (sections 2.7 and 3.7) and
   [MODEL_IO.md](../spec/MODEL_IO.md) (section 3), and the checkpoint and row-cache layout from
   [STRATA_MEMORY.md](../spec/STRATA_MEMORY.md).
4. Gives each card half of every layer ([ADR-0023](../adr/0023-two-cards-tensor-split.md)), subtracts the fixed rows
   and the reserve for N sequences from 24,385,683,456 B, and divides what is left by half the mean expert,
   1,023,200 B.

Units: GB is 10^9 B and GiB is 2^30 B. Bytes are exact where the row is *computed*.

## Weights

### Per layer (computed)

Source: [model/tensors-IQ3_S.tsv](../model/tensors-IQ3_S.tsv), summed per layer. "Dense" is every non-expert tensor of
the layer except the PLE tensors (layer 1 only, 65,740,800 B; below). "Per expert" is the layer's three expert tensors
divided by 512: (gate + up + down) / 512. The types are the file's ([the tensor list](../model/tensors-IQ3_S.tsv)).

| Layer | Kind | Dense B | Gate and up / down type | Expert tensors B | Per expert B |
| ---: | --- | ---: | --- | ---: | ---: |
| 0 | GDN | 75,710,336 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 1 | GDN | 76,283,776 | IQ2_S/Q2_0 | 773,324,800 | 1,510,400 |
| 2 | GDN | 75,692,416 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 3 | attn | 62,789,632 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 4 | GDN | 80,443,776 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 5 | GDN | 72,077,696 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 6 | GDN | 75,692,416 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 7 | attn | 66,414,592 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 8 | GDN | 71,163,776 | IQ2_S/Q2_0 | 773,324,800 | 1,510,400 |
| 9 | GDN | 75,710,336 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 10 | GDN | 76,334,976 | IQ2_S/Q2_0 | 773,324,800 | 1,510,400 |
| 11 | attn | 68,127,232 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 12 | GDN | 72,633,216 | IQ3_XXS/Q2_0 | 878,182,400 | 1,715,200 |
| 13 | GDN | 68,951,936 | IQ2_S/Q2_0 | 773,324,800 | 1,510,400 |
| 14 | GDN | 73,570,176 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 15 | attn | 67,267,072 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 16 | GDN | 75,270,016 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 17 | GDN | 79,765,376 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 18 | GDN | 73,590,656 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 19 | attn | 72,481,792 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 20 | GDN | 79,765,376 | IQ3_XXS/Q2_0 | 878,182,400 | 1,715,200 |
| 21 | GDN | 77,676,416 | IQ3_S/Q2_0 | 956,825,600 | 1,868,800 |
| 22 | GDN | 76,132,736 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 23 | attn | 72,220,672 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 24 | GDN | 79,970,176 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 25 | GDN | 80,187,776 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 26 | GDN | 75,090,816 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 27 | attn | 71,708,672 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 28 | GDN | 80,661,376 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 29 | GDN | 76,488,576 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 30 | GDN | 75,710,336 | IQ3_XXS/Q2_0 | 878,182,400 | 1,715,200 |
| 31 | attn | 76,142,592 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 32 | GDN | 78,150,016 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 33 | GDN | 72,187,776 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 34 | GDN | 71,993,216 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 35 | attn | 76,360,192 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 36 | GDN | 76,606,336 | IQ2_S/Q2_0 | 773,324,800 | 1,510,400 |
| 37 | GDN | 76,388,736 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 38 | GDN | 74,194,816 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 39 | attn | 75,988,992 | IQ2_S/IQ4_NL | 1,009,254,400 | 1,971,200 |
| 40 | GDN | 79,970,176 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 41 | GDN | 69,003,136 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 42 | GDN | 72,159,616 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 43 | attn | 76,833,792 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 44 | GDN | 72,279,936 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 45 | GDN | 72,661,376 | IQ3_S/IQ4_NL | 1,192,755,200 | 2,329,600 |
| 46 | GDN | 72,579,456 | IQ3_XXS/IQ4_NL | 1,114,112,000 | 2,176,000 |
| 47 | attn | 77,012,992 | IQ4_XS/IQ4_NL | 1,363,148,800 | 2,662,400 |
| Sum | | 3,576,097,280 | | 50,292,326,400 | mean 2,046,400 |

Worked example for layer 3: gate and up are IQ2_S, 268,697,600 B each, so 2 x 268,697,600 / 512 = 1,049,600 B per expert;
down is IQ4_NL, 471,859,200 / 512 = 921,600 B; together 1,971,200 B. The per-expert size runs from 1,510,400 B
(layers 1, 8, 10, 13, 36) to 2,662,400 B (layer 47). The mean, 50,292,326,400 / (48 x 512) = 2,046,400 B, is about
489 experts per GB.

### The parts outside the layers (computed)

| Part | Bytes | Source |
| --- | ---: | --- |
| Token embedding (IQ4_XS) | 337,715,200 | `token_embd.weight` row of the `.tsv` |
| Head and final mixer | 534,620,160 | `output.weight` (Q6_K) 521,472,000 + `output_hc_down` 6,553,600 + `output_hc_up` 6,553,600 + `output_hc_norm` 40,960 |
| PLE, layer 1 (BF16, F16, F32) | 65,740,800 | the six `blk.1.ple_*` rows |
| N-gram table (IQ4_NL, on the SSD) | 28,800,138,240 | `per_layer_token_embd.weight`; not held in RAM or VRAM ([ADR-0015](../adr/0015-ngram-table-on-the-ssd.md)) |
| Draft (MTP) layer, BF16 | 5,214,301,696 | [MODEL_FILES.md](../MODEL_FILES.md#the-mtp-layer): not in the GGUF |
| of which its 512 experts | 5,033,164,800 | `gate_up_proj` 3,355,443,200 + `down_proj` 1,677,721,600; 9,830,400 B per expert |
| of which everything else | 181,136,896 | 5,214,301,696 - 5,033,164,800 |

The 48 layers' dense and expert bytes plus the embedding, the head, the PLE and the n-gram table
add to the file's 83,606,638,080 B: 3,576,097,280 + 50,292,326,400 + 337,715,200 + 534,620,160 + 65,740,800 +
28,800,138,240.

### What one decode token reads (computed)

Method: the bytes of every tensor one decode token reads, summed from [the tensor list](../model/tensors-IQ3_S.tsv),
with 10 of 512 experts per layer. Arithmetic, not a measurement.

| Part | Bytes per token |
| --- | ---: |
| Dense weights: projections (~1.95 GB), hyper-connections `hc_*` in BF16 (1.26 GB), routers (0.13 GB), shared experts (0.15 GB) | 3.58 GB |
| Routed experts, 10 per layer x 48 (~2.05 MB each) | 0.98 GB |
| The head (`output.weight`, Q6_K) | 0.53 GB |
| Per-layer embeddings | 0.07 GB |
| Weights in all | ~5.1 GB |
| Recurrent state, 36 DeltaNet layers | 0.11 GB read, 0.11 GB written |
| Selected KV, 12 attention layers | ~0.025 GB |
| The indexer's keys | ~0.006 GB at 4K, ~0.2 GB at 128K, ~0.4 GB at 262K |

About 6.7B parameters are active per token: 2.36B in routed experts, 0.24B shared, ~3.4B dense, 0.64B in the head.
Both cards read their half of every layer at once, so one token reads ~2.6 GB per card. The dense weights, not the
experts, are most of the bytes, so the dense kernels' rate counts as much as the experts'.

## Per sequence

All sizes are for the engine's 8-bit KV (Q8_0, 32 values in 34 B; [ADR-0008](../adr/0008-kv-cache.md)).

### KV (computed)

[MODEL_LAYER.md section 3.7](../spec/MODEL_LAYER.md#37-the-kv-layout-and-what-is-stored-per-token), 12 attention layers:

- K and V: 12 x (1,024 / 32) x 34 = 13,056 B per token, 1,088 B per layer.
- Indexer keys: 12 x (128 / 32) x 34 = 1,632 B per token, 136 B per layer.
- Together 14,688 B per token.

| Context | All KV in VRAM | Streamed: K and V in VRAM | Streamed: indexer keys in VRAM | Streamed: VRAM total | K and V in RAM (streamed) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 60,162,048 | 53,477,376 | 6,684,672 | 60,162,048 | 53,477,376 |
| 32,768 | 481,296,384 | 427,819,008 | 53,477,376 | 481,296,384 | 427,819,008 |
| 131,072 | 1,925,185,536 | 427,819,008 | 213,909,504 | 641,728,512 | 1,711,276,032 |
| 262,144 | 3,850,371,072 (3.59 GiB) | 427,819,008 | 427,819,008 | 855,638,016 (0.80 GiB) | 3,422,552,064 (3.19 GiB) |

How the streamed columns are made:

- K and V in VRAM = 13,056 x min(context, 32,768). Only the 32,768 most-read cells are resident
  ([STRATA_MEMORY.md section 5](../spec/STRATA_MEMORY.md#5-the-kv-cache)).
- Indexer keys in VRAM = 1,632 x context. The indexer scores *every* cached token to pick the 2,048 it attends
  to ([MODEL_LAYER.md](../spec/MODEL_LAYER.md#37-the-kv-layout-and-what-is-stored-per-token)), so its keys cannot be
  streamed on demand. Strata keeps its pooled keys "always in VRAM" ([STRATA_MEMORY.md](../spec/STRATA_MEMORY.md#5-the-kv-cache)),
  and this budget does the same. Streaming the indexer keys too would save 427,819,008 B of VRAM per 262K sequence.
- K and V in RAM = 13,056 x context: the whole authoritative copy, as in Strata.

At 262,144 tokens the all-in-VRAM figure is [MODEL_LAYER.md's](../spec/MODEL_LAYER.md#37-the-kv-layout-and-what-is-stored-per-token)
3,850,371,072 B, which is ADR-0008's "~3.6 GB".

### State and the rest (computed)

| Item | Bytes | Arithmetic or source |
| --- | ---: | --- |
| Recurrent state, per GDN layer | 3,268,608 | conv 3 x 10,240 x 4 + recurrence 128 x 128 x 48 x 4 ([MODEL_LAYER.md 2.7](../spec/MODEL_LAYER.md#27-state-per-sequence)) |
| 36 GDN layers | 117,669,888 | 36 x 3,268,608 |
| PLE conv history, layer 1 | 368,640 | 9 x 10,240 x 4 (this is the "n-gram history"; the two previous token ids add 8 B) |
| All recurrent state of one sequence | 118,038,528 | 117,669,888 + 368,640 |
| Draft layer KV, 32,768-cell window | 35,651,584 | 32,768 x 1,088 ([MODEL_IO.md section 3](../spec/MODEL_IO.md#3-the-draft-mtp-layer)); it attends to the last 32,768 cells only, at any context |
| A conversation checkpoint | 118,038,528 | the same running state ([STRATA_MEMORY.md section 8](../spec/STRATA_MEMORY.md#8-conversation-checkpoints)); Strata says ~118 MB, and the sum agrees. Strata's checkpoint also keeps each attention layer's small indexer tail, whose size is not in the specs, so this is a floor |
| A parked conversation at 32K | 581,509,120 | checkpoint 118,038,528 + K and V 12 x 1,088 x 32,768 = 427,819,008 + draft window 35,651,584 |
| A parked conversation at 262K | 3,576,242,176 | checkpoint + K and V 12 x 1,088 x 262,144 = 3,422,552,064 + draft window |

The draft layer needs only its 32,768-cell ring, so this budget keeps no more of it in RAM than the window. Strata
counts 13 layers over the whole context (13 x 1,056 B per token).

## Per card, for the tensor split

[ADR-0023](../adr/0023-two-cards-tensor-split.md): each card holds half of every layer. Every row is *computed* from the
rows above unless labelled; the two cards are the same.

### The fixed rows, per card

| Row | Bytes | Label |
| --- | ---: | --- |
| Usable VRAM | 24,385,683,456 | measured ([the device memory record](MEASUREMENTS.md#2026-10-11-device-memory)) |
| Dense weights, half of 3,576,097,280 | 1,788,048,640 | computed |
| PLE (layer 1), half | 32,870,400 | computed |
| Token embedding, half its columns | 168,857,600 | computed. It could live in host memory instead: one row per token |
| Head and final mixer, half the vocabulary | 267,310,080 | computed |
| Draft layer, half | 614,446,848 | *estimate*, which predates the choice of the draft layer's storage. The core keeps its experts in IQ4_NL, 707,788,832 B a card for them alone (the `ENGINE_MTP_CACHE_DIR` file, [DEVELOPING.md](../DEVELOPING.md)) |
| Working memory | 1,600,663,296 | *estimate* |
| Left for the reserve and the experts | 19,913,486,592 | |

### The reserve per sequence, per card

| Context | K and V (one head, at most 32,768 cells) | Indexer keys (half the blocks) | Recurrent state (half the heads) | Draft KV (half) | Per sequence |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 26,738,688 | 3,342,336 | 59,019,264 | 17,825,792 | 106,926,080 |
| 32,768 | 213,909,504 | 26,738,688 | 59,019,264 | 17,825,792 | 317,493,248 |
| 131,072 | 213,909,504 | 106,954,752 | 59,019,264 | 17,825,792 | 397,709,312 |
| 262,144 | 213,909,504 | 213,909,504 | 59,019,264 | 17,825,792 | 504,664,064 |

### What is left for experts

Each cached expert takes half its bytes on each card, 1,023,200 B at the file's mean of 2,046,400 B, so both cards
hold the same count: floor(free / 1,023,200).

| Sequences x context | Free per card | Experts cached | Share of 24,576 | Not cached |
| --- | ---: | ---: | ---: | ---: |
| 1 x 32K | 19,595,993,344 | 19,151 | 77.9% | 5,425 |
| 2 x 32K | 19,278,500,096 | 18,841 | 76.7% | 5,735 |
| 4 x 32K | 18,643,513,600 | 18,220 | 74.1% | 6,356 |
| 1 x 262K | 19,408,822,528 | 18,968 | 77.2% | 5,608 |
| 2 x 262K | 18,904,158,464 | 18,475 | 75.2% | 6,101 |
| 4 x 262K | 17,894,830,336 | 17,489 | 71.2% | 7,087 |

Spot check: 1 x 262K: 19,913,486,592 - 504,664,064 = 19,408,822,528; / 1,023,200 = 18,968.8, so 18,968.

### RAM for the tensor split

Pinned means Level Zero host memory (`zeMemAllocHost`), which both cards and the CPU read. Streamed KV per sequence
is K and V for the whole context plus the draft window: 13,056 x context + 35,651,584.

| Sequences x context | Arena: every expert | Arena: only the experts not cached | Streamed KV | Pinned, every expert | Pinned, not cached only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 x 32K | 46.84 GiB | 10.34 GiB | 0.43 GiB | 47.27 GiB | 10.77 GiB |
| 4 x 32K | 46.84 GiB | 12.11 GiB | 1.73 GiB | 48.56 GiB | 13.84 GiB |
| 1 x 262K | 46.84 GiB | 10.69 GiB | 3.22 GiB | 50.06 GiB | 13.91 GiB |
| 2 x 262K | 46.84 GiB | 11.63 GiB | 6.44 GiB | 53.28 GiB | 18.07 GiB |
| 4 x 262K | 46.84 GiB | 13.51 GiB | 12.88 GiB | 59.72 GiB | 26.39 GiB |

The arena of experts not cached is (24,576 - cached) x 2,046,400 B, from the table above. Beside the pinned memory:
the n-gram row cache (0.09-1.41 GiB for 1M-16M rows of 90 B), checkpoints (118,038,528 B each in this plan; about
4.25 GiB each in the core at 262K, [record](MEASUREMENTS.md#2026-10-16-engine-start-and-quit)) and the two Python
processes (the engine's measured 0.40-0.42 GiB RSS at `READY`, same record; the front end's not measured). These are
ordinary memory, not large contiguous pages. Level Zero's host allocations do not count against the user's lock limit
([the host memory record](MEASUREMENTS.md#2026-10-12-host-memory)).

Measured on the host on 2026-10-13, with nothing else on the cards ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)):

| Measure | Value |
| --- | --- |
| `MemAvailable` | 51.5-55.2 GiB |
| Host memory allocated in 256 KiB pieces before the free 2 MiB blocks ran out | ~12 GiB |
| Allocated before the host began to swap (compaction and page-cache reclaim from ~12 GiB on) | ~25 GiB |
| The same probe on 2026-10-12, before swapping | 45 GiB |

One or two sequences with the not-cached arena (10.8-18.1 GiB) fit inside the range where the host only compacts
and reclaims page cache. Four at 262K (26.4 GiB) reach the edge. An arena of every expert does not fit.
