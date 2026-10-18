# Results

What each measurement of the engine and its cards found. Each entry below is one measurement: the question it
answered and its answer, newest first within its area. The code, the ADRs and the other pages link these entries for
their numbers. The full records, with the method, the software versions and the state of the machine for each run,
are kept privately, because together they identify the machine they were taken on.

## The test machine

Every number here was measured on one machine:

- two Intel Arc Pro B60 cards (24 GB each), each on its own PCIe 5.0 x8 link to the CPU;
- an AMD Ryzen 9 7945HX: 16 Zen 4 cores on two CCDs, with AVX-512 (VNNI, VBMI, BF16, IFMA);
- 96 GiB of dual-channel DDR5 at 5200 MT/s, about 90 GiB of it visible to Linux;
- a PCIe 4.0 NVMe SSD, which holds the model files;
- Linux 6.17 with the in-tree `xe` driver, and the engine image's user-space stack: compute-runtime 26.35.39758.10,
  IGC 2.41.5 and the Level Zero loader 1.32.0 ([ADR-0002](adr/0002-pin-the-toolchain.md)).

Other software shares the machine. A number that rests on the host (the CPU's experts, host memory, load times)
moves with what else runs, so compare such numbers only within one entry.

## How the numbers were taken

- A number is the median of at least five runs, given with its range (least to most).
- A change is judged against the code before it in ABBA order: rounds of A, B, B, A, at least three, so each arm
  runs six times. Each run is a fresh process with the same inputs
  ([ADR-0016](adr/0016-how-a-change-is-judged.md)). The two arms of one entry ran in one sitting; an entry's A arm
  is not the previous entry's B arm, so gains from several entries do not add up.
- A kernel is timed after at least 50 ms of work on its card. An idle B60 sits near 1200 MHz, and a kernel timed
  there looks 1.7-2 times slower ([kernel spike](#2026-10-11-kernel-spike)).
- A change that alters tokens says so. Otherwise the arms gave the same tokens.
- `doc-4k-qa`, `doc-32k` and the other named prompts are in `reference/prompts.jsonl`. `long-4k`, `long-32k`,
  `long-128k` and `long-256k` are longer prompts of the measurements' own, not in the repository.

## Current figures

| Figure | Value | Record |
| --- | --- | --- |
| Decode of doc-4k-qa, ms per token (median of the record's B arm) | 17.255 | [2026-10-15-sub-bytes-words](#2026-10-15-sub-bytes-words) |
| Prompt of doc-4k-qa (4,159 tokens), tokens/s (median of the record's B arm) | 2,135 | [2026-10-15-prompt-router](#2026-10-15-prompt-router) |
| `greedy_test` against llama.cpp, prompts of up to 512 tokens: positions that agree | 1,046 of 1,063 | [2026-10-15-matvec-k-split](#2026-10-15-matvec-k-split) |

## The machine and its stack: cards, host memory, Level Zero, the image

### 2026-10-16-vram-give-back

Can the engine give its VRAM back without exiting, and map it again at the same addresses so its recorded lists still work?

Device memory yes, up to 16 GiB a card; 20 GiB frees in 3.0-3.6 ms and maps again in 159-160 ms, but a list recorded before the free then crashes the driver (SIGSEGV in libze_intel_gpu) once more than 16 GiB is mapped. Freed VRAM is back in Sysman's figure at once, or within 0.1 s if the other card read it. Host physical memory fails with OUT_OF_DEVICE_MEMORY. An idle process with the runtime and all kernels holds 200 MiB of VRAM a card.

### 2026-10-16-peer-exchange

Is the decode exchange faster with each card writing its partial sums into the other card's memory than through host memory?

Yes at four rows, by less at one. In the engine's form (an event per exchange, the other card's list waiting on it), 96 exchanges of 2,560 floats a row cost 3.9-5.0 µs each through host memory at one row and 10.2-10.3 at four; with the kernel writing into the other card, 2.3-3.4 and 5.1-5.2. That saves 0.15 ms a step at one row and 0.48 at four, every sum checked. A copy into the other card saves almost nothing at one row and 0.43 ms at four. ADR-0023 ruled peer-to-peer out; ADR-0030 later allowed it.

### 2026-10-13-engine-image

Does the root Dockerfile build, pin ADR-0002's versions and give a non-root container both cards and what the build needs?

Yes. The image builds with nothing copied to the host, every version matches ADR-0002, and as user ubuntu a container sees both B60s, compiles for bmg-g21, has an unlimited memory lock and reads /models.

### 2026-10-12-host-memory

Can host memory from zeMemAllocHost hold the expert arena for both cards, and how fast do the cards read it?

Both cards read it from one context at 28.3 GB/s each (~45 GB/s together); the lock limit does not apply; past 26-32 GiB the host swaps, so the 46.84 GiB arena does not fit; importing existing memory does not work.

### 2026-10-12-card-to-card

How long do the two B60s take to signal each other and exchange partial sums through host memory?

A tensor-parallel exchange at 1-4 tokens takes 10.0-10.5 µs, 96 in a row 0.99-1.03 ms; the flags-only round trip is 6.4 µs, with a 205 KB hand-off 50.3 µs. One fp16 store per work-item is 13-54x slower and gave wrong sums.

### 2026-10-11-level-zero-report

What does the Level Zero driver report on each B60 (extensions, queue groups, copy engines, atomics)?

Every extension the design uses is listed; one context holds both cards; one copy engine per card (ordinal 1); host memory is ATOMIC but not CONCURRENT_ATOMIC; DPAS and DP4A flags set; Sysman gives no PCIe speed.

### 2026-10-11-launch-cost

What do kernel launches cost, does a recorded layer replay, and how do per-token values reach recorded kernels?

960 small kernels take 1.28 ms launched one by one and 0.83 ms replayed as 48 recorded lists (0.06 ms of host time); per-token values go through memory; mutable lists and indirect launches work.

### 2026-10-11-kernel-spike

Are OpenCL C kernels enough for the expert and dense products, and what share of the card's bandwidth do they reach?

Yes (ADR-0003 confirmed); experts read 257-344 GB/s at one token (56-76% of 455), dense 370-421 GB/s; the reordered layout is 3.2-8.8x faster; the card needs ~50 ms of work to leave 1200 MHz.

### 2026-10-11-doorbell

Does a spinning GPU kernel see the CPU's write to host memory on Xe2, and what does a round trip through host memory cost?

Only with a system-scope acquire fence before each volatile load (100 of 100); an atomic acquire load sees it in 21-37 of 100. The per-layer round trip is 4.54 µs median, 5.88 µs p99; events work at about the same cost.

### 2026-10-11-device-memory

How much device memory can one allocation and one card take, is past 4 GB addressed, and what read bandwidth does a card reach?

One allocation takes 22.71 GiB, 23.00 GiB with relaxed limits; a card holds 25.31 GB, and past VRAM allocations spill to RAM instead of failing; only Sysman's free figure shows VRAM's end; reads reach 455 GB/s a card, 909.5 GB/s for both.

### 2026-10-11-cpu-bar

How fast is ggml's CPU backend on this model's expert products on the 7945HX, and is it bound by arithmetic or memory?

i-quants run 5.0-5.7 GB/s a core and 40-44 GB/s on 16 cores (arithmetic-bound); IQ4_XS and IQ4_NL hit ~50 GB/s already on 8 cores (memory-bound); Q2_0 is scalar at 2.3 GB/s a core; ggml gets no reuse across 2-4 tokens.

## Design inputs: hit rates, prefetch, the decode and prompt forecasts, references

### 2026-10-17-video-pair-reference

What are the encoder's reference rows for a pair of video frames, and how close does the engine's encoder come to them?

llama.cpp's CPU encoder (flash attention off) and an F64 computation of the spec agree on two frame pairs to a least cosine of 0.9991 at 64 tokens and 0.9966 at 252; with the frames swapped the least falls to 0.34. The engine on two B60s agrees with llama.cpp to a mean cosine of 0.99991-0.99992 and a least of 0.9962-0.9983, and takes 38.5-40.9 ms a pair. transformers' video processor and the front end sample, size and time test-3.mp4 alike.

### 2026-10-17-sleep-and-wake

How long do a sleep and a wake take with the arena kept and the weights read with O_DIRECT, what does a sleeping engine hold, and does a load still fill the page cache or swap?

A sleep took 376-718 ms (median 393) and a wake 14.2-20.7 s (median 15.0), 11.9-18.1 s of it reading the cards' 38.30 GB; greedy tokens after every wake equalled those before. Asleep, the engine held 0.31-0.32 GiB of VRAM a card. A start took 24.5-64.7 s to READY, 19.9-54.3 s of it the load, which read 58.95 GB. The page cache rose by at most 0.79 GiB across any load. A wake swapped out 0-50 MiB of the host, a start 2.7-790 MiB, and the slow starts were those with the most compaction stalls. The front end's lease, lapse and /admin/sleep worked end to end.

### 2026-10-17-requests-sent-together

Why do four short requests sent at once sometimes reach their first tokens a second later than in other runs, and does waiting a moment after a request before planning stop it?

The engine process planned after reading only some of four GEN lines sent within 0.1 ms in 10 of 51 runs, and the requests it had not read reached their first tokens 0.55-1.78 s later than in the arm's other runs; waiting up to 2 ms after a GEN to an idle engine, 0 of 24 runs did. Against the engine before the change, six runs an arm, a lone request's first token is 2,158 ms against 2,150; of four short requests sent together, the last first token is 3,191 against 3,190 (the base's range 3,124-4,122); and four 4,093-token requests finish 9.9% sooner, from a different prompt order.

### 2026-10-17-kv-streaming

Would KV streaming (ADR-0008 item 3), with 32,768 resident cells per attention layer and the rest in host memory, make the engine significantly faster in some case without slowing the usual one much?

Not on this host. The VRAM it frees caches 32-40 more experts per layer (307 to 339 with the draft layer at one 262,144-token slot), which makes deterministic decode 1.5-3.9% faster (60.85 to 62.71 tokens/s over three prompts) and is lost in acceptance changes with drafting (77.32 to 73.97 over seven prompts); two slots at 131,072 gain nothing (19.93 against 20.19 ms a step). Past 32K a CLOCK cache of 8,192 pages a layer misses 5.8-10% of selected pages, and copying them costs 0.10 ms a step; reading the KV from host memory in place costs 50% of a decode step and 5-6% of a prompt.

### 2026-10-17-encoder-reference

What are the image encoder's reference rows, and how close does the engine's encoder come to them?

llama.cpp's flash attention (taken for this head size by its F16 one-row path) moves rows far from an exact computation, mean cosine 0.957 at 4,096 tokens; without it llama.cpp and an F32 computation agree to a least cosine of 0.990. The engine's encoder on two B60s agrees with the F32 computation to a mean cosine of 0.9998-0.99995 and a least of 0.960-0.998, as close as llama.cpp without flash attention. 4,096 tokens take 1.75-1.76 s on two cards.

### 2026-10-16-load-read-rates

How fast can a load read the model from the SSD with O_DIRECT, and how fast does the CPU rearrange the experts into the cards' layout?

O_DIRECT reads the 51 GiB first shard at 6.49-6.64 GB/s from 8 threads of 1 MiB on a quiet host, about 9.2 s for the 59.8 GB a load needs; to_runs on 16 threads rearranges at 12.26-13.04 GB/s, about twice that, so the CPU does not limit a load.

### 2026-10-16-load-phases

Where does an engine load's minute go, phase by phase, and does a load push the host into swap?

A cold load at a 262K context took 63.2-71.5 s (median 64.4), 55.4-63.0 s of it the experts; the arena's 20.49 GiB is copied out of the mapped file on one thread in 39.4-46.2 s; recording every list takes 0.19 s and the warm-up chunk 1.9 s. The draft layer adds 42.8 s, nearly all quantising on the CPU. The load as it was then swapped 2.4 GiB of the host's other memory out; with pages given back per layer, 11-53 MB.

### 2026-10-16-engine-start-and-quit

How long do the engine child and the front end take to start, how fast does QUIT give the cards and the RAM back, and what does the engine pin at its production defaults?

QUIT gave both cards' VRAM back 0.44-0.64 s after the line was sent, and the child exited after 0.61-1.11 s. The child reached READY in 86.0-94.2 s, 8.7-8.9 s of it Python reading the GGUF twice before engine_open. The front end alone takes 5.3-5.6 s to serve, 4.4 s of that the same GGUF read. At the defaults (8 checkpoints) the child pins about 20.5 GiB of arena plus about 4 GiB per checkpoint, more than the test host had free; the one such run was stopped by its memory guard.

### 2026-10-14-expert-prefetch

Does copying likely misses to a card ahead of time make decode faster, and how well does each signal predict the misses?

No. An emulated prefetch took 23.88 ms per token against 21.83 (ABBA, 6 runs each); the next layer's router finds 54% of misses at 10 guesses; the copies slow the cards' kernels 5-12%. The engine does not prefetch.

### 2026-10-13-design-inputs

What do a decode step on two cards, the host's pinned memory, shared bandwidth, the Python-C++ boundary, step lists, timing and the prompt cost?

Both cards on every layer with the CPU on the misses give 9.8-11.3 ms per token (one card 14.9-15.6); a 2,048-token chunk ~2,150 tokens/s; the host gives pinned memory by contiguous pages, ~12 GiB clean on 2026-10-13.

### 2026-10-12-expert-hit-rates

What share of routed experts do the cards hold, and how often does a layer have no miss, from routing traces of the reference prompts?

At K = 24 with 17,581 experts, a static cache hottest first serves 89.6% of routed experts, Strata's swaps 95.9%; at T = 3, 26.9% of layers have no miss (46.1% with swaps).

## The decode step's kernels

### 2026-10-17-paged-kv

Does reading the KV cache through a page table kept by Python (ADR-0033) cost decode or prompt time?

No measurable cost. doc-32k decode 18.62 against 18.48 ms per step, 76.10 against 75.24 tokens/s with drafting, prompt 2,243.5 tokens/s in both (ABBA, 6 runs an arm); the table's reads add about 11 µs to an 18.3 ms step in traces; the same tokens in every run.

### 2026-10-16-peer-exchange-decode

Does a decode window exchanging its partial sums through the cards' own memory, under ADR-0030, make decode faster, and would a flag in the other card's memory beat the event?

By 0.9-2.0%. With the mixer's out product and the MoE block's combine writing into the card's memory and one copy placing them in the other card's, for two rows or more, decode went from 72.19 to 72.88 tokens/s at long-4k, 74.97 to 75.62 at long-32k, 70.83 to 71.50 at long-128k and 69.55 to 70.98 at long-256k, the tokens per step unchanged. At long-32k the median is inside the base's range. ADR-0030 adopted the change. In a probe, a flag in the other card's memory in place of the event saved 0.20 ms a step of 96 exchanges at one row, 0.16 at two, and cost 0.05 at four.

### 2026-10-16-out-products-host-writes

Do the mixer's out products (ssm_out, attn_output) lose time writing their partial sums into host memory in-kernel, and does writing them in the card's memory and copying them out give it back?

Yes at two rows and more, no at one. Written into host memory a float at a time, a Q6_K out product took 24.8 µs at two rows and 35.6 at four, against 21.4 and 26.3 written in the card's memory and copied out; at one row 19.9 against 20.0. With the copy for windows of two rows or more, decode went from 69.79 to 71.06 tokens/s at long-4k, 71.94 to 73.28 at long-32k, 68.62 to 69.93 at long-128k and 67.76 to 69.22 at long-256k, the tokens per step unchanged. At two rows card 0 spent most of the time saved waiting longer for the CPU's experts.

### 2026-10-16-more-experts-cached

How many more experts do the cards cache at a 262,144-token context with the prompt path's indexer scores sized for one card and 512 MiB left free, and what does it give?

288 to 324 experts per layer without the draft layer (267 to 303 with it); prompt 2,474.6 to 2,555.4 tokens/s on long-4k and 2,117.1 to 2,166.6 on long-32k, decode 52.55 to 54.76 on long-4k and, in one process, 52.9-53.1 to 54.5-54.9 on long-32k; each card still had 1.31 GiB free by Sysman after a run.

### 2026-10-16-indexer-select

Why does the indexer's top-513 selection take 155 µs a launch at 256K, and what takes it down?

idx_cand 1.865 to 0.92 ms a decode step at 246K positions; one launch at 30,834 values 80-279 to 59-155 µs; long-128k's prompt 1,956.2 to 1,994.6 tokens/s and decode 63.48 to 64.58, long-4k unchanged. Merging hot bins alone was not enough.

### 2026-10-16-host-glue

How long do the cards wait for the host in a decode plan, on which parts of the host's work, and what does reading a step's n-gram rows at once give back?

Card 0 waited for the host 0.73 ms a plan at long-4k, under the 0.8 ms bar, so the host does not hold up each plan as supposed; but most of the host time it waited on was layer 0 waiting for the n-gram rows, read row by row. Reading a step's rows at once and dequantising them on the calling thread took layer 0's excess from 0.91 to 0.19 ms a plan at long-4k and from 2.18 to 0.25 at long-256k. Decode went from 70.63 to 71.61 tokens/s at long-4k, 70.32 to 74.49 at long-32k, 69.86 to 71.05 at long-128k and 69.10 to 70.41 at long-256k, the tokens per step unchanged.

### 2026-10-16-gpu-clock-in-decode

Does the GPU clock drop below its top during decode, between plans or while a card waits for the CPU, and do plans at a lower clock take longer?

No. Read by Sysman every 1.17 ms through 1,024 decoded tokens of long-4k, long-32k, long-128k and long-256k with the draft layer and swaps, both cards were at 2,400 MHz, the top of their range, in 99.74-100% of samples, with no throttle reason; one plan of 2,444 saw a clock under 2,350. The clock never came near 1,200 MHz; waiting between plans it fell to 2,250 for about 2 ms at most.

### 2026-10-16-drafting-and-swaps

Should the engine process turn on drafting with the MTP layer and adaptive swaps, and do they cost prompt speed?

Yes. Decode went from 54.76 to 65.31 tokens/s on long-4k and from 50.31 to 66.99 on long-32k. The prompt lost 2.1% and 1.6% against the same engine without them (2,555.4 to 2,500.9, 2,166.6 to 2,132.2 tokens/s), the draft layer's memory, and stayed above the engine before more experts were cached (2,474.6 and 2,117.1).

### 2026-10-16-draft-rule-simulated

Would another draft rule (draft count, probability threshold, cumulative probability, skipping catch-ups) decode faster?

No. Replayed on 1,022 positions of long-256k with measured costs, the engine's rule (up to 3 drafts, each at probability 0.5 or more) gave 64.01 tokens/s; the best of 270 others 64.34, 0.5% more.

### 2026-10-16-draft-rounds

What keeps drafting with the MTP layer from paying at the model's sampling, and do the draft path's three fixes help?

Each verify-window row costs about 5 ms (70 more CPU experts); drafts sampled as the request samples, chain drafts at the catch-up's cell and q and k norms +1 took decode from 54.33 to 58.45 tokens/s on long-4k and 54.40 to 57.89 on long-32k, 1.579 to 1.645 tokens per step.

### 2026-10-16-draft-layer-xmx

How much of a draft-layer pass is its BF16 products on the older kernels and its plan read from host memory, and what does running them as the decode step does give back?

Confirmed, and the change trades. On mv_xmx_bf16 and hc_up_mix_xmx with its plan on the card, a draft-layer pass took 0.15 ms less at one row, 0.23 at two and 0.45 at four, and acceptance on the same text held (1.691 tokens a plan against 1.688). Decode went from 73.54 to 76.67 tokens/s at long-32k, 70.03 to 71.83 at long-128k and 69.00 to 70.44 at long-256k, but from 70.69 to 69.92 at long-4k, where tokens per step fell from 1.655 to 1.624. With its rounding moved and one length slower, it was not adopted.

### 2026-10-16-draft-layer-passes

How much cheaper can the draft layer's passes and a round's commit be without fewer drafts accepted, and does a smaller draft head pay?

A round's commit and catch-up from 2.42 to 1.97 ms and a chain draft from 1.39 to 1.34 ms, the same drafts; decode 65.31 to 65.51 tokens/s on long-4k and 66.99 to 67.15 on long-32k, within the runs' spread. A smaller draft head made a chain draft 0.98 ms but fewer drafts were accepted (0.749 against 0.798 of first drafts), and it was not kept.

### 2026-10-16-decode-launches-together

Does every decode launch waiting for the one before cost time, and do the independent products gain more launched together or fused into one launch?

Yes. With the products that share inputs launched together in a list not in order, a plain step at long-4k went from 17.52 to 16.74 ms; through the C ABI decode medians of 4 runs went from 61.38 to 67.52 tokens/s at long-4k and 67.87 to 71.38 at long-32k, while long-128k read 65.43 to 62.93 with two of the change's runs slowed by the host's swap, the same tokens per step. Fusing two products into one launch beat launching them together by only 1.4-2.4 µs a pair.

### 2026-10-16-bf16-products-occupancy

Are the decode step's BF16 products short of work-groups, does splitting K over work-groups help, and what does reading the step plan from host memory cost a launch?

Not short of work-groups. Each launch at one row is 1.8-2.9 µs over a streaming read of its bytes, and with its second launch K split over work-groups was 0.5-1.9 µs slower than unsplit. A word read from host memory at a launch's start costs 0.8 µs; with the plan copied to each card first, a plain step went from 19.00 to 18.62 ms, and decode 1.8-2.3% faster at long-4k, long-32k and long-128k, the tokens per step unchanged.

### 2026-10-15-sub-bytes-words

What does bytewise subtraction in whole 32-bit words give the Q6_K, Q2_0 and i-quant decoders?

About 4% of the Q6_K products and half a percent of a step; the Q6_K loop 187 instructions against 299; decode 17.34 to 17.255 ms per token, ranges overlapping; the prompt within noise.

### 2026-10-15-router-top10

Where does the one-row router kernel spend its 14.6 µs, and what do a threshold top 10 and a work split give?

The top 10 (3.9 µs) and the job written by one sub-group (6.3 µs); now one row takes 5.3 µs; router_group 0.421 against 0.685 ms of card 0's step; decode 17.88 to 17.67 ms per token.

### 2026-10-15-router-group

What does decode gain from running the router and the expert grouping in one launch?

router_group took 0.688 ms of card 0's step against 0.822 ms for the pair; decode 19.17 to 19.07 ms per token, ranges overlapping.

### 2026-10-15-matvec-k-split

Are the decode step's products of up to 512 rows faster with K split over 4 sub-groups?

Yes. A launch 3.5-4.6 µs against 6.7-7.7; card 0's five products 0.81 against 1.22 ms; decode 17.76 to 17.48 ms per token; greedy agreement 1,046 of 1,063.

### 2026-10-15-indexer-selection

Why do the indexer's idx_cand and idx_merge take 37 and 57 µs, and how much does decode gain without it?

Host-memory stores and a serial search, not the selection; one launch of each went from 37.0 to 9.8 µs and 57.1 to 15.7 µs; decode 20.32 to 19.56 ms per token; prompt 2,069 to 2,088 tokens/s.

### 2026-10-15-hc-down-split

Is a K split of 8 better than 16 for the hyper-connections' down product?

Yes. 18.1 against 19.1 µs a launch for one row; decode 18.07 to 18.03 ms per token, ranges overlapping; the rounding moved greedy agreement to 1,041 of 1,063.

### 2026-10-15-gdn-fusion

What does decode gain when the DeltaNet layer's norm and gates move into gdn_conv and gdn_recur?

Two launches a layer go away; the kernels took 0.78 ms of card 0's step against 0.98; decode 18.04 to 17.85 ms per token; the same bits.

### 2026-10-15-bf16-products-xmx

Are the decode step's BF16 products faster on the XMX units with bf16 activations?

Yes. The router's product 9.0 µs against 12.1, the PLE key product 123 against 169 µs; card 0's five products 1.41 to 1.11 ms; decode 19.13 to 18.82 ms per token; misses 25.01 to 25.12 a token from the rounding.

### 2026-10-15-a8-fusion

What does decode gain when the kernels that make the a8 activations write them, in place of quantize_a8?

Five launches a layer go away, quantize_a8's 0.69 ms a card is gone for 0.15 ms more elsewhere; decode 19.35 to 19.11 ms per token, ranges overlapping.

### 2026-10-14-topk-rows

How much of topk_rows' 281 µs does a select that ranks only a few candidates give back?

topk_select_few takes one row of 124,160 logits in 26.2 µs instead of 281.3 µs, the same result ties included; decode 20.69 to 20.19 ms per token.

### 2026-10-14-decode-expert-kernels

Were the decode step's grouped expert products limited by memory or by instructions, and what does a rewrite give?

By instructions. moe.cl with 32-bit offsets and vector loads takes 2.0-2.4x less time for gate and up and 1.2-2.1x for down; card 0's two kernels 3.96 to 2.29 ms; decode 22.16 to 20.72 ms per token.

## The prompt path's kernels and its reads

### 2026-10-17-swaps-keep-the-copies

Why does each request in one process reach its first token later than the one before with the anonymous arena and swaps, and does keeping every staging slot's arena piece across swaps stop it?

Each swap round recorded the prompt path's copy lists again, and from anonymous memory the runtime binds every copy's source when a list is recorded (21-46 us) and unbinds it when the list is freed (50-89 us); as swaps split the runs of adjacent experts the copies grew from 198 to 4,740 a card, and a 512-token request went from 911 to 2,232 ms. With the outgoing expert copied into the incoming one's piece the list is recorded once, and s512 takes 488 against 815 ms, s4352 1,549 against 2,418, long-4k 1,472 against 2,565, long-32k 13,237 against 13,638 and long-128k 50,680 against 51,076 as one request after the warm-up, the same tokens.

### 2026-10-17-staging-half-downs

Does a card staging only its half of every uncached down make prompts faster, and by how much at 64 to 112,981 tokens?

Confirmed. A card's staged bytes fall from 14.45 to 10.07 GB a chunk and its copy list's busy time by 27% (656 to 481 ms at 64 rows); the time to the first token falls 25-32% for 64 to 1,024 tokens, 3-5% at 4,018 to 31,338 tokens, and 0.7% (inside the spread) at 112,981; the region copy runs at half rate, so the arena keeps each down as two regions.

### 2026-10-17-prompt-windows

Are prompt pieces of up to 64 tokens faster read as decode windows than as prompt chunks, and by how much after a restore at 4k, 32k and 128k positions?

Yes. Through the C ABI a 1-token prompt reaches its first token in 19.7 ms against 465.9, 16 tokens in 77.8 against 466.4, 64 in 330.5 against 483.5, and one token after a restore in 17.0-17.6 ms against 463.8-504.2; after restores at 32,768 and 131,072 positions 1 token takes 21.9-23.2 ms and 16 tokens 88-118 ms against 462-504; 65 tokens and more are unchanged. On a later base (paged KV, the arena in driver memory) 1 token takes 20.7 ms against 457.5, 64 tokens 325.7 against 452.7, one token after a restore 16.5-17.3 against 434-455, and 65 and 128 tokens are 2.4% and 4.9% slower.

### 2026-10-17-prefill-timings

Where does a prompt's time go on the engine as it stands, from 64 tokens to 246,674, kernel by kernel, with the copy engine, the SSD, the CPU, checkpoints and the cards' clocks?

A chunk of 64 to 1,024 tokens takes 650-765 ms, of which 166-484 ms is the cards waiting for the 14.45 GB of uncached experts each card stages per chunk at 19-22 GB/s; a 4,352-token chunk takes 1,645-1,680 ms (2,590-2,646 tokens/s) at position 0 and 2,748 ms at position 239,360, where an attention layer costs 132 ms against 41 ms; a 246,674-token prompt reads in 136.0 s (1,813.6 tokens/s); checkpoints save and restore at 15-180 ms.

### 2026-10-17-ngram-ahead

Does reading the next prompt chunk's n-gram rows while this chunk runs make prompts of several chunks faster?

Yes, by 1.3-1.5%. Through the C ABI long-32k reaches its first token in 14,078 ms against 14,289 on one base (-1.5%) and 14,481 against 14,698 on a later one (-1.5%), long-128k in 55,082 against 55,561 (-0.9%) and 55,381 against 56,086 (-1.3%), with the same tokens; prompts of one chunk are unchanged.

### 2026-10-17-idx-score-rows

Does p_idx_score reading each pooled key once for eight rows make long prompts faster, and does it change any score?

Yes at 128k, no change at 32k and below. The kernel takes 20.7 ms a layer in place of 49.0-52.7 for 4,352 rows at position 239,360 and 6.1 in place of 7.1 at 20,000, every score bit-identical; through the C ABI long-128k reaches its first token 2.5% sooner on one base (54,157 against 55,561 ms) and 2.9% sooner on a later one (54,482 against 56,086), with the same tokens; long-32k and the 4k cases are within their spread.

### 2026-10-17-hc-split

If each card computes the hyper-connections for half of a prompt chunk's rows and sends the other card its half of the mixed x, is a chunk faster, and do the tokens stay the same?

Yes from 2,048 rows. At 4,352 rows, sending x by 128-byte block stores costs 0.61 ms on top of the half rows' 0.43, against 0.85 for every row on each card (2-byte stores take 11.7 ms). Through the C ABI s4352 reaches its first token 6.2% sooner (1,559 against 1,663 ms), long-4k 5.8%, long-32k 5.0% (13,651 against 14,366) and long-128k 5.0% (51,110 against 53,800), with the same tokens; chunks of 1,024 rows and fewer do not split and are unchanged.

### 2026-10-17-hc-lowrank

Do other gemm.cl tiles make the hyper-connections' two low-rank products faster in a prompt chunk, and does that show in the time to the first token?

For the down product only. With 32 tokens by 32 rows a sub-group and 11 sub-groups a work-group, one work-group covers its 324 columns, and the product takes 0.775-0.782 ms in place of 0.911-0.917 for 4,352 rows and 0.180-0.222 in place of 0.293-0.351 for 1,024, bit for bit; it is slower at 256 rows and fewer. Through the C ABI long-32k and long-128k reach their first token 0.5% sooner (14,353 against 14,422 ms, 53,741 against 54,019) and s1024 1.8% sooner, with the same tokens. No tile beats the up product's 32 by 32.

### 2026-10-17-anon-arena-default

Does the anonymous arena cost anything against the driver's arena (prompt rate at 64, 1,024 and 4,352 tokens, decode, load, tokens), and can its cost on short prompts be removed?

Its first form made 64-token prompts 17-21 ms slower, as each of the prompt path's 17,712 copies a card costs about 1.3 us more from anonymous memory; with region copies (copy_rows) it beat main's driver arena, 64-token prompts 282.8 against 294.2 ms over 3 ABBA rounds, with 1,024 and 4,352 tokens and decode the same, loads no slower in a load-only ABBA, and the same tokens in every run.

### 2026-10-16-swaps-and-prompts

Why do adaptive swaps slow the prompt path, and what keeps them from it?

A swap round makes the prompt path record its copy list again, about 35,000 copies, before its next chunk; 45 ms on a 4K chunk of 1,552, 8 ms with the list recorded on a thread of its own. With rounds also held while a prompt is read, long-32k's prompt went from 2,088.8 to 2,132.2 tokens/s with swaps and the draft layer; long-4k's did not move (2,499.7 and 2,500.9).

### 2026-10-15-prompt-router

How much does the prompt path's router gain with one sub-group per token?

A launch for 4,159 tokens 29.3 µs against 987 µs; the router 1.54 ms of each card's prompt against ~48 ms; prompt 2,087 to 2,135 tokens/s.

### 2026-10-15-prompt-moe-scatter-blocks

What bounds p_moe_scatter, and what do sub-group block reads and writes give?

Its stores; with block reads and writes it takes 0.77-0.79 ms (271-276 GB/s) against 2.02; 33.6 ms of each card's chunk against 98.4; prompt 2,594 to 2,698.5 tokens/s.

### 2026-10-15-prompt-moe-scatter

Is scattering each token's row once faster than gathering the experts' activations?

Yes. A launch for 4,159 tokens 2.03 ms against 2.74; 98.5 ms of each card's chunk against 131.8; prompt 2,135.5 to 2,174.5 tokens/s.

### 2026-10-15-prompt-gemm-tiles

Are the prompt path's bf16 products faster with tiles of 4 tokens where K or N is short?

Yes, the same bits. Down 884 µs against 1,129, up 783-803 against 955, ab 48.5 against 77-82; prompt 2,261.5 to 2,351.5 tokens/s.

### 2026-10-15-prompt-gemm-bands

Are the prompt path's products with weights larger than L2 faster when taken in bands of rows?

Yes, the same bits. qkv 1.86 against 2.20 ms, q 2.40 against 2.62, PLE keys 3.82 against 7.66; nothing below L2; prompt 2,417 to 2,505 tokens/s.

### 2026-10-15-prompt-gdn-fused-loop

Is the prompt path's Gated DeltaNet recurrence faster with one loop and one barrier a token?

About 5%. 2.46 ms against 2.60 alone; 88.8-89.3 ms of each card's chunk against 93.6-94.3; prompt 2,685 to 2,708.5 tokens/s. Faster forms that change the sums lost greedy agreement.

### 2026-10-15-prompt-expert-down

What bounds the prompt path's expert down product, and what do early loads, block writes and half tiles give?

No one part; with all three IQ4_NL takes 2.84 ms against 3.59 and Q2_0 2.62 against 2.97; 148.9-149.1 ms of each card's chunk against 178.7; prompt 2,545.5 to 2,599 tokens/s.

### 2026-10-15-prompt-expert-block-fields

What bounds the prompt path's expert gate and up product, and does loading each block's shared fields once help?

The decoding's loads. With block_raw, IQ2_S takes 5.66 ms against 7.27 and IQ3_S 5.74 against 7.15; 284.1 ms of each card's chunk against 329.4; prompt 2,216 to 2,275.5 tokens/s.

### 2026-10-15-prompt-exchange-stores

Can the prompt path's kernels write their partial sums into the other card directly, in place of a copy?

Yes, when each store is at least 64 bytes (22.2-22.5 GB/s, against 1.9 GB/s for one fp16 a work-item); p_hc_combine_norm 136-139 ms of a chunk against 243-246; prompt 2,353.5 to 2,427 tokens/s.

### 2026-10-15-prompt-dequant-stores

Is dequant_f16 faster for the prompt path's dense weights with packed words and whole-line stores?

Yes. A launch 50-113 µs against 246-501; 10.45 ms of each card's chunk against 46.9; prompt 2,173 to 2,212.5 tokens/s (12 runs each).

### 2026-10-15-prompt-attention-mask

How much of the prompt path's tiled attention is its mask, and what does computing it once a tile give?

About a quarter; 7.16 ms against 9.02-9.03 on a 4,159-row chunk; 84.5 ms of each card's chunk against 106.5-106.8; prompt 2,509 to 2,545 tokens/s. Tiles of 32 keys wait on greedy_test.

### 2026-10-15-ngram-gather

Does gathering a prompt's n-gram rows on the CPU hold up the prompt, and do more threads or Linux AIO read them faster?

Usually not; doc-4k-qa's 66,544 rows took 37.6 ms with an empty cache; 256 threads 31.2 ms against 33.6 from 128; io_uring is refused in the container and Linux AIO took 76.9-84.8 ms.

## The CPU's experts and the arena

### 2026-10-17-decode-stalls-eviction

What takes the arena's pages away during decode, so that steps stall for seconds while the host reclaims memory?

xe moves the engine's host buffers (12.2 GB, the arena among them) from GTT to system placement under the host's reclaim; decode stalled 6.92 s and 2.75 s while it moved them, then every expert's first read faulted (566,270 major faults over the 5,415 steps after).

### 2026-10-17-anon-arena-under-reclaim

Does the anonymous arena stop the decode stalls under the host's reclaim, can the copy engines read it, and do buffers xe moves keep their contents?

Yes to all three. Under one 54.8 GB page-cache read during decode the driver arena stalled 5.75 s and the anonymous arena's worst step was 48.1 ms; both cards' copy engines read anonymous memory at 28.4 GB/s, as from zeMemAllocHost; every run gave the same 4,000 tokens, the moved buffers' run included.

### 2026-10-17-anon-arena

Does an arena in the process's own memory (Options anon_arena) decode as fast as the driver's, with the same tokens?

Yes, 16.54 and 16.61 ms per step against 16.58 and 16.56 ms (ABBA, two runs an arm), the same 2,000 tokens in all four; the driver's host buffers fall from 9.07 GB to 0.06 GB (both cards' fds) and the arena's 8.05 GiB moves into the process.

### 2026-10-16-cpu-answer-per-layer

Is a window's time bound by the CPU's latency per missed layer, or by its throughput per missed expert, and how long does card 0 wait for the CPU in a step of 1 to 4 rows?

Throughput. A missed layer's answer took 20-25 µs plus 60-62 µs per distinct expert and 19-25 µs per further row on an expert already in the job, and a step's cpu_wait tracked its distinct experts (R2 0.80-0.84) more than its missed layers (0.42-0.66). Card 0 waited 2.89, 4.31, 6.09 and 8.11 ms a step at 1, 2, 3 and 4 rows on long-4k, 16-26% of the step and 4.5 ms a plan.

### 2026-10-15-cpu-one-job

Does compute_many in one pool job make the CPU's misses faster?

One expert's call 73.7-75.0 µs against 80.9-81.3; decode 18.23 to 18.08 ms per token, ranges overlapping; the same bits.

### 2026-10-15-cpu-miss-path

Does compute_many in two pool jobs, and workers that poll between jobs, make the CPU's misses faster?

Two jobs, yes, one expert 80-85 µs against 88-93; decode 18.34 to 18.15 ms per token. Polling 5 ms between jobs made the mean and tail steps worse, so workers sleep.

### 2026-10-15-arena-touch

Why are some of the CPU's expert calls slow, and does reading the arena once at the end of load fix them?

An expert's first miss faulted, 268 µs with ~60 major faults against 88 µs; reading one byte a page of the arena at load removes the faults; decode 19.12 to 18.44 ms per token; card 0's cpu_wait 2.13 to 1.25 ms.

### 2026-10-14-decode-stalls

Why do a few decode steps take 0.3-9 s while the host is short of memory?

The CPU's share of the experts stalls on ~60 major page faults an expert, its arena pages unmapped; it needs memory pressure; giving back the file's page cache removed most. What unmaps the pages is not established.
