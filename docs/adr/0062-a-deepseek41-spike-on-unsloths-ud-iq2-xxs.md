# ADR-0062: a deepseek41 spike on Unsloth's UD-IQ2_XXS file, its experts in VRAM, an arena and the SSD

**Status:** draft, 2026-11-01; not accepted. For `deepseek41` only, it departs
from items 2-4 of [ADR-0006](0006-scope.md) (item 2 as
[ADR-0056](0056-the-aim-is-the-fastest-performance-on-battlemage.md) items 5-7 restate it), items 1, 2 and 5 of
[ADR-0023](0023-two-cards-tensor-split.md), item 7 of
[ADR-0032](0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md) where it keeps the file mapped for swaps,
items 1 and 2 of [ADR-0012](0012-correctness-reference.md),
[ADR-0054](0054-the-bf16-checkpoint-is-the-end-to-end-reference.md), item 6 of
[ADR-0057](0057-the-engine-reads-the-host-it-runs-on.md) and [ADR-0044](0044-a-short-prompt-is-read-whole.md), and
amends item 3 of [ADR-0015](0015-ngram-table-on-the-ssd.md). Flash-Next's rules do not change.
**Revisit when:** the measured hit curve, CPU rates and SSD rate show that splitting by layers would remove more
miss time than the dense time it adds; the exchanges under the misses' and SSD reads' load cost more than the dense
time the tensor split saves; a step stalls past the wait bound; or the SSD tier's reads are measured to bound decode
more than another host layout would.

## Context

- DeepSeek-V4.1-Flash-1017 is GGUF architecture `deepseek41` ([#28](https://github.com/avra-m3/c0ven/issues/28)).
  It has 40 layers of 384 experts, 6 routed and 1 shared a token, each expert 5,120 x 2,304. Its 4 residual streams
  are mixed by hyper-connections. Attention has one 512-wide KV head, a window of 128 positions, and from layer 2 on
  a compressed cache (2 positions a row at layers 2-19, 1 at 20-39) read through an indexer's top 512. Layers 1 and 14
  add Engram rows: 24 rows a layer of a 768,022,850-row table, hashed from the last 4 tokens.
- Unsloth's UD-IQ2_XXS file is 300.5 GB in 6 shards. Its experts are IQ1_S, IQ2_XXS and IQ2_S (gate and up) and
  Q2_K, IQ2_XS, IQ2_XXS, IQ2_S and IQ3_XXS (down), all in 256-value blocks: 146.1 GiB in all, a mean expert of
  8.49 MiB at layers 0-19 and 10.99 MiB at 20-39. The dense tensors and the head are 7.18 GiB. The Engram table is
  Q5_K, one 176-byte block a row, 135,172,021,600 B in the last shard.
- Split by tensors, each card holds 4.04 GiB of dense weights and its part of about 104 experts a layer at layers
  0-19 and 80 at 20-39; the cached experts are 34.1 GiB across both cards (*estimate*, from the file's tensor sizes,
  the same bytes for every layer, one slot at 16,384 positions, 0.75 GiB of working memory and 0.33 GiB of recorded
  lists a card). About 112 GiB of experts stay off the cards, more than the host's 92 GB of RAM
  ([ADR-0006](0006-scope.md) item 1).
- The pinned llama.cpp `3cf03257f` has no `deepseek41`. Upstream llama.cpp runs it since
  [#31086](https://github.com/ggml-org/llama.cpp/pull/31086) and its follow-up
  [#31137](https://github.com/ggml-org/llama.cpp/pull/31137), with the stream in F32 and no simulation of DeepSeek's
  FP8 and FP4 activations. The BF16 checkpoint needs a multi-GPU host far larger than the one used for Flash-Next.
- Each layer has its own window KV. Compressed rows and index keys come from layer 2 for layers 2-7, 8 for 8-13, 14
  for 14-19 and 20 for 20-39. The indexer's top-k comes from the same source at layers 2-19 and 20-23, and from 24,
  28, 32 and 36 for 24-27, 28-31, 32-35 and 36-39.

## Decision

1. **A second architecture, `deepseek41`, is built from Unsloth's UD-IQ2_XXS file.** It is a spike: correct tokens
   first, with Flash-Next's speed rules applied from the first version.
2. **deepseek41 splits by tensors as ADR-0023 does, except as this item says.** Each card computes half the query
   heads, half the output groups and half the columns of `attn_output_b`. A routed expert splits by rows into whole
   256-value blocks: part A is 1,280 of its 2,304 rows, part B the other 1,024. Each card's cache for a layer has as
   many part-A slots as part-B slots, so its count is even, and a cached expert holds part A on one card and part B on
   the other, as its slot pair sets. The shared expert splits 1,152/1,152 where it is Q8_0 and 1,280/1,024 where it
   is Q5_K. The KV
   head, the compressed rows, the router, norms, hyper-connection mixes, compressor and indexer are whole on both
   cards, and the indexer's keys are not split by position. The head splits by vocabulary rows and Engram's product
   by rows. Each Engram layer adds a third exchange to the layer's two.
3. **Experts are held in three tiers, each given as a size.** The cards hold a hot set per layer, sized in bytes and
   filled hottest first from a routing profile. An arena of the engine's anonymous memory, under
   [ADR-0035](0035-the-arena-in-the-engines-own-memory.md) and [ADR-0037](0037-the-cards-vms-without-page-faults.md),
   holds the next-hottest, sized by the memory plan from the host's available memory less pinned memory, the working
   set and a headroom. The rest stay on the SSD. The fit report gives each tier's experts from the core's own counts,
   its expected share of routings and the SSD reads a token. Experts never go through the page cache: the shards' file
   pages are given back after load, and only `token_embd` stays mapped. This departs from ADR-0023 item 5, under
   which every expert is on the cards or in the arena, and from ADR-0032 item 7's mapping for swaps.
4. **The CPU computes every miss** with ggml's kernels, from the arena or from a small anonymous staging ring, in one
   pool job a layer that adds the arena experts' sum and then the SSD experts' sum, each in routing order. The SSD
   tier is read with `O_DIRECT` on pool threads: each layer's SSD misses in one read round, each expert's three
   matrices at once, never on the engine thread.
5. **Every wait on the CPU, the other card or the Engram rows is a bounded spin** under
   [ADR-0011](0011-cpu-gpu-signalling.md)'s rules. It gives up after 2.4e9 ticks of the device clock, which counts
   core cycles: at least 1 s, and at most 2 s at clocks of 1,200 MHz or more. It then sets a stuck flag, and the step
   fails. A list waits on an event that the host signals only when a host deadline thread also holds that event and,
   1 s after the list reaches it, sets the stuck flag and signals it. The miss wait comes after the card's shared and
   cached experts, so they run while the CPU answers.
6. **The correctness reference is upstream llama.cpp at `a074215352c382e66a108597bcf7f4f31e00d111`,** the merge of
   #31137, its CPU backend, with f16 caches, on the same file. The end-to-end gate is a mean KL of at most 0.04
   against it. This departs from ADR-0012 items 1 and 2 and from ADR-0054 for this model; ADR-0012 items 3 and 4
   stand.
7. **The Engram reads follow ADR-0015,** with rows of 176 B, issued as soon as the step's token ids are known. The
   card waits for both layers' rows before layer 1 and copies them to the card. This amends ADR-0015 item 3, which
   names layer 1 as the only reader. The table is never read into RAM: ADR-0057 item 6 does not apply to a 135 GB
   table, and `--ngram-ram` has no effect on this model.
8. **The KV and indexer state stays on the cards, indexed by position, and is priced in experts.** The context is at
   most 16,384 positions until the indexer's candidate filter is built; below that the filter keeps every block.
   Before the limit is raised, 8-bit compressed rows are measured against f16 by `kl_test`.
9. **Prompts are read as decode windows until a prompt chunk path exists,** and afterwards only below a crossover
   measured on this model. The chunk path stages only the experts its router picks
   ([ADR-0046](0046-a-short-chunk-fetches-the-experts-its-router-picks.md)) below a measured cut-off. A chunk is
   submitted as layer-range lists that each end well inside xe's 5 s job timeout, with writers' steps between them.
   ADR-0044's pieces do not apply to this model until then.
10. **The generic `StepIn` carries a third earlier token, `p3`,** which the 4-token Engram hash needs. Flash-Next
    ignores it.
11. **The arithmetic follows upstream llama.cpp:** the stream in F32, the window and compressed caches in f16, the
    activations of BF16 products in bf16, no FP8 or FP4 activation simulation. A build switch rounds the stream to
    bf16 as DeepSeek's code does. [ADR-0016](0016-how-a-change-is-judged.md)'s rule and switches apply unchanged.

## Why

- ADR-0023's reason holds: both cards work on every token. Split by layers at 20, the cards would cache about 133
  more experts, but read 7.71 GB of dense weights a token one card after the other against 4.34 GB on both at once:
  about 8-9 ms more a token at 370-420 GB/s
  ([measurements](../measurements/MEASUREMENTS.md#2026-10-14-decode-expert-kernels)), against about 3 ms of misses
  the extra experts remove at uniform routing (*estimate*, from the file's sizes, the SSD read rate
  ([measurements](../measurements/MEASUREMENTS.md#2026-10-16-load-read-rates)) and an assumed CPU rate of
  32.7 GB/s). The tensor split pays 82 exchanges a token for this; it keeps its lead while each costs under about 65 us
  on average (*estimate*, by the same figures).
- Cutting every expert at the same block and taking the larger part from the slot keeps both cards' bytes equal, so
  no cached expert is lost to the unequal parts, and any expert fits any slot pair.
- A page-cache host tier lets the file's reads reclaim the host's memory, the condition under which decode stalled
  for seconds on Flash-Next; an anonymous arena held the worst step to 48.1 ms under the same reclaim
  ([measurements](../measurements/MEASUREMENTS.md#2026-10-17-anon-arena-under-reclaim)).
- A card reading a missed expert in place took about 565 us against about 61 us on the CPU, for a 0.94 MiB expert
  ([measurements](../measurements/MEASUREMENTS.md#2026-10-19-pcie-share-of-misses)).
- Whether a spin held past xe's 5 s job timeout resets the engine is not measured
  ([LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md) row 29); a 6.92 s step with card 0 in its CPU wait finished
  normally ([measurements](../measurements/MEASUREMENTS.md#2026-10-17-decode-stalls-eviction)). A bound of 1-2 s
  keeps every wait well inside 5 s and leaves room for an SSD read round of hundreds of MiB. A list's event wait has
  no bound on the card, so a host thread gives it one.
- Adding the CPU's sums in a fixed order keeps the tokens the same from run to run and between one slot and several.
- Upstream llama.cpp is the only engine that runs this file, so it is the only reference for greedy tokens and KL.
  DeepSeek's own code is the reference for each operation, with tolerances.

## Options not chosen

- **Splitting by layers at 20:** experts stay whole, one hand-off a step instead of 82 exchanges, the same bits on one
  card and two. The dense layers then run one card after the other (above), and one card cannot hold the model
  within the test machine's memory, so the same bits on one card buy nothing.
- **The larger part chosen by the expert's id and layer:** a fixed slot must then hold the larger part on both cards,
  about 10% fewer cached experts, and a swap could only replace an expert of the same choice.
- **Down split exactly 1,152/1,152 inside a block:** it needs a half-block runs layout for 5 types and an edge case in
  each down kernel. It is the follow-up if the measured imbalance passes 2% of the step.
- **The page cache over the file's mapping as the host tier:** above.
- **DeepSeek's code as the token reference:** it simulates FP8 and FP4 activations that no GGUF engine computes, so
  tokens could not agree even with correct kernels.
- **A prompt chunk path at once:** decode windows give correct tokens with no new kernels; the chunk path follows the
  first end-to-end decode.

## Consequences

- deepseek41 has its own `core/models/deepseek41/`, `kernels/models/deepseek41/` and `engine/models/deepseek41/`,
  and one line in `core/models/open.cpp` ([ADR-0061](0061-the-core-picks-its-model-by-the-files-architecture.md)).
- Helpers shared with qwen4exp are copied during the spike and lifted into shared code afterwards, as pure moves
  with qwen4exp's results unchanged.
- `ModelMemory` gains per-card weights, the Engram cache, the staging rings, and the host and SSD tiers as sizes
  ([ADR-0060](0060-the-control-plane-takes-a-models-rules-through-modelsetup.md)); the fit report prints them and a
  serving-time watch reports host swap-out.
- The CPU's down product reads two unequal column parts, and h's task rows are a multiple of the down type's dot
  block. The expert kernels read each group's slot class from a per-layer table.
- Speed depends on the host's available memory and on the SSD, which the fit report gives (ADR-0057).
- A file on a filesystem without `O_DIRECT` is refused, as ADR-0015 item 5 has it.
