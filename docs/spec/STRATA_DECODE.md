# Strata's design, part 1: the decode loop

> Under [ADR-0001](../adr/0001-level-zero-from-scratch.md), Strata is the reference design, not the code base. This
> page describes its decode loop as behaviour, with what each part was measured to be worth. No Strata code is
> copied: the page uses prose, tables and diagrams only ([AGENTS.md](../../AGENTS.md)).

**Source.** [avra-m3/strata-intel](https://github.com/avra-m3/strata-intel) at commit
`1678de333d0e0711bc414ad992b640e1a37dd814` (`1678de3`, 2026-10-02, engine 0.1.34). Paths below (`src/...`,
`include/...`, `serve/...`, `docs/...`) are in that tree. The sources are the code named in each section,
`docs/DETAILS.md`, `docs/HOW_IT_WORKS.md`, `docs/paper/Strata-Paper.pdf` (the paper), and the commit messages that
carry a measurement.

**Where the numbers come from.** Strata measured on its "reference PC", unless a row says otherwise: an RTX 5070
12 GB on PCIe 4.0 x16, a Ryzen 5 7600 (6 cores, AVX-512), 64 GB DDR5-5200, an NVMe SSD, and Windows. No number here
comes from a B60 or the 7945HX. [Part 2](STRATA_MEMORY.md) uses the same machine. "Not measured" means neither the docs, the paper nor a
commit message gives a number for it. Strata is CUDA/HIP: "graph" below means a captured CUDA graph, and "mapped
memory" means pinned host memory the GPU reads and writes through PCIe.

**Geometry used for sizes** (`include/strata/kernels/ngram.hpp`, `cpu/expert.hpp`, `qsa.hpp`): 48 layers, 36 Gated
DeltaNet (GDN) and 12 sparse-attention (QSA); n_embd 2560; 4 hyper-connection streams; 512 routed experts per
layer, top 10, plus one shared expert; the QSA indexer pools one key per 4 cells (`idx_block` 4, key width 128).

## In short

| # | Mechanism | Where | Measured effect |
| --- | --- | --- | --- |
| 1 | The verify window: up to T tokens of one sequence through all 48 layers in one graph | `src/core/verify.cpp`, `include/strata/core/verify.hpp` | 2.4-3.6 tokens per window; MTP speculation 1.6-1.8x decode |
| 2 | Drafting: MTP draft layer, suffix lookup, draft policy | `src/core/mtp.cpp`, `src/spec/` | lookup: code edits +6-11%, other text unchanged |
| 3 | Acceptance and rollback | `src/program/generate.cpp` (serve loop), `verify.cpp` `commit` | exact: identical to greedy, tested with forced wrong drafts |
| 4 | One layer: doorbell, flags A/B/M, E-6 skip, combine | `verify.cpp` `record_window` / `run`, `src/core/expert_source.cpp` | E-6 neutral (off by default); PCIe share +10-16% for i-quants |
| 5 | Per-sequence state, carved per layer range, no token-path allocations | `include/strata/core/session.hpp` | not measured |
| 6 | Sampling | `verify.cpp` `run`, `src/kernels/cuda/sampler.cu`, `include/strata/kernels/sampler.hpp` | penalties cost 1-11% decode |
| 7 | Stall watchdog (#29, #31, #267) | `generate.cpp`, `include/strata/core/progress.hpp`, `verify.cpp` | not measured (correctness); a polled final wait cost ~3% and was reverted |
| 8 | The commit: replay accepted rows from stored inputs | `verify.cpp` `capture_commit` / `commit` | async commit +5.6% decode (IQ2_XS) |
| 9 | Identical tokens: a multi-token kernel's per-token arithmetic is the single-token kernel's | `verify.hpp`, kernel headers | switches for byte-identical repeats: 1 of 4 answers differ vs 2 of 4 |
| 10 | The server: line protocol, endpoints, FIFO, conversation parking | `serve/server.py`, `generate.cpp` `--serve` | not measured (parking) |

## 1. The verify window

**What it does.** One decode round puts T tokens of one sequence at consecutive positions p .. p+T-1 through all 48
layers in one pass: the last accepted token (row 0) and T-1 draft tokens. The head runs on every row, and row t's
pick is the token plain decoding would produce after row t's token. The dense weights are read once for all T rows.
That is why the window pays off: reading ~3.5 GB of dense weights is most of the GPU's cost, and it barely grows with
T (paper §3.3, §5).

**How it runs.**

- Each window size T has its own graph, captured on first use (`Verifier::capture`, up to 9 slots for T = 0..8). The
  first window of a request has T = 1, so a session ends up holding graphs for every size it used (the
  RTX 5090 log in `bench/results/2026-09-30-community-rtx-5090/engine.log` captures sizes 6, 1, 4, 3, 2).
- Per window the host stages, in mapped memory: the T token ids, per-row step records and per-head positions, and
  the T n-gram (PLE) rows. The PLE rows are hashed from the tokens and the last two tokens before the window, and
  read from the SSD-backed table synchronously before the launch (`Verifier::run`).
- The host then launches the graph once and runs the per-layer loop of section 4 against it.
- Hard limits: `kVerifyMaxT` = 8 (`verify_kernels.hpp`), the CPU multi-token expert kernels' `MAXT` = 8, at most 128
  routed entries per window in the CPU dispatch tables (`expert_source.cpp`).

**Data it keeps** (`Verifier::init`): one device arena sized for the largest window, carved once ("57.1 MiB of device
buffers" for a 6-token window, IQ2_XS, RTX 5090 log above), and mapped staging: tokens, steps, positions, PLE rows
(T x 2560 floats), the doorbell payload (T x 2560 floats of MoE input, T x 10 ids, T x 10 weights), the three flag
words and the sequence counter, the GPU plan block, and the CPU's result rows (T x 10 x 2560 floats, 614 KB at T = 6).
It also keeps 16 VRAM staging slots for the PCIe share of the misses (`kStagingBlobs`), each the size of the largest
expert blob.

**How its size is chosen.** Per round, in the serve loop (`generate.cpp`, the decode `while` loop):

| Setting | Default (setup's config) | Effect |
| --- | --- | --- |
| `--spec T` | 4 | the MTP's window: up to 3 drafts |
| `--suffix-draft N` | 3 (on) | turns `--spec 4` into an MTP cap of 4 (`--mtp-max-t`) and a window cap of 6: a suffix-lookup window may be 2 tokens longer |
| `--spec-min-p P` | 0.5 | the window holds row 0 plus the leading drafts whose draft probability is at least P |
| first window of a request | - | T = 1 (no drafts exist yet) |
| near the context end | - | the loop stops when p + T would pass `--max-context` |

So with defaults T is 1 + (number of leading MTP drafts with p >= 0.5), at most 4, unless the draft policy picks a
lookup window of up to 6 (section 2).

**Measured.**

- Tokens per window (accepted drafts + 1), paper table 4: Q2_0 2.71 / 3.23 / 3.11 / 2.80 / 2.65 / 2.46 at
  1K / 4K / 32K / 64K / 128K / 262K; IQ2_XS 2.42-3.28; IQ3_XXS 2.42-3.63. The docs summarize it as 2.4-3.2.
  Long contexts accept fewer drafts (paper finding 12).
- Speculation with the MTP layer is worth 1.6-1.8x: plain decoding 47-57 tokens/s at 4K, with the window 82-92
  (paper finding 2). Less than the 2.5-3x of all-in-VRAM engines, because every extra row routes to its own CPU
  experts.
- What a wider window costs: a window's union of distinct experts is 1.75x / 2.4x / 3.05x one token's at T = 2 / 3 / 4,
  from decode traces (`verify.hpp`). The draft policy's prior: a round costs 1.35 / 1.7 / 2.05x one token's at
  T = 2 / 3 / 4, "+10 ms per token with every missed expert on the CPU" (`src/spec/draft_policy.cpp`, citing
  `bench/results/2026-09-27-spec`, which is not in the tree at this commit).
- One window at 4K, Q2_0: 34.3 ms = GPU 14.6 + waiting for CPU experts 13.4 + drafting 2.0 + other 4.3, at 3.23
  tokens per window and a 0.72 VRAM hit rate (paper table 5). IQ2_XS 42.0 ms, IQ3_XXS 49.4 ms.
- The split window (`--spec-split`, off): cutting a window into two token groups so one group's CPU experts overlap
  the other's GPU work is exact but ~7% slower, 85.8 vs 91.8 tokens/s at 4K (paper finding 5).

## 2. Drafting

Three parts: the MTP draft layer (the main source), the suffix drafter (prompt lookup), and the draft policy that
chooses between them each round. A fourth, `strata::spec::Controller` (`src/spec/controller.cpp`), is a cost-model
chooser with per-position acceptance estimates; at this commit only its unit test uses it.

### The MTP draft layer (`include/strata/core/mtp.hpp`, `src/core/mtp.cpp`)

- **What it is.** The model's one multi-token-prediction layer, fetched from the BF16 checkpoint (the GGUF files do not
  carry it; ~5 GB downloaded) and packed with Q8_0 projections and Q2_0 experts (`tools/mtp_pack.py`,
  `tools/mtp_rt.py`). Cell i pairs the main model's final multi-stream residual at position i with the token at i+1,
  and predicts the token at i+2. Its own residual feeds the next draft step.
- **Its shape at run time.** Embedding and hidden projections, the sum into the four streams, one attention
  hyper-connection, attention over its own K/V, an MLP hyper-connection, a full MoE (all 512 experts resident in VRAM,
  ~675-708 MB, run by the grouped hit kernels), the final mixer and a head.
- **Simplifications that touch only draft quality.** Dense attention over every cell it has seen, within a window of
  the last 32,768 cells (`--mtp-window`). Below 2,051 cells this equals the model's sparse selection. So speculative
  cells never touch indexer state and are overwritten when their positions are processed again.
- **The draft head** covers a token subset (`mtp/rt/draft_vocab.bin`): 106,299 ids by default ("cjk", ~138-180 MiB),
  40,525 for "en", 58,963 for "cyrillic". A draft can only be a token of the subset.
- **VRAM.** ~0.9 GB with its K/V and buffers (835 MiB in the RTX 5090 log), plus the draft head. It is loaded before
  the expert cache is sized.

**One round** (`MtpDrafter::draft`), run after the window's commit:

1. A *round graph*: catch-up over the window's T cells (row t = the window's final residual at p+t paired with the
   window's pick at row t), then draft 0 from row a (the last accepted row). One graph launch and one stream sync.
2. Then one *step graph* per further draft, each a launch and a sync, while the last draft's probability under the
   draft layer is at least `--spec-min-p`, up to the MTP cap minus one.
3. Out: the drafts and each draft's probability (the softmax probability of the draft head's argmax).

Each draft's probability comes back to the host, and the next window's size depends on them (section 1). The prompt path
fills the drafter's K/V in batches (`MtpDrafter::prefill`), skipping cells the 32K window can never reach.

**Measured.** Acceptance of greedy drafts at steps 1-3 in the transcription check: 0.89 / 0.86 / 0.85
(`mtp.hpp`, `tools/mtp_probe.py`). Drafting costs ~2 ms of a ~34-49 ms round (paper table 5). The CJK subset made
CJK answers 15-38% faster; the Cyrillic one took Russian/Ukrainian from 1.4 to 2.1 tokens per round, 83 -> 109
tokens/s on an RTX 5090 fork (DETAILS.md). The 0.5 floor's own effect is not measured in the docs. `tools/calibrate.py`
measures 0.3 / 0.5 / 0.7 per PC and keeps a value only if it beats 0.5 by 3%.

**Coupled draft sampling** (`include/strata/core/coupled_draft.hpp`, `STRATA_SPEC_COUPLED=1` or `--coupled-draft`,
off by default). For a sampled request the drafter samples with the target's chain and the *same* Philox draw the
verify row will use (counter = MTP cell + 1), instead of proposing its argmax. Verification is still exact-match
against the target's sample, so the text does not change. Effect: not measured (the commit says "acceptance should
rise").

### The suffix drafter (`include/strata/spec/suffix_drafter.hpp`, `src/spec/suffix_drafter.cpp`)

- **What it does.** Proposes the tokens that followed the longest earlier occurrence of the sequence's current
  suffix. It costs microseconds and needs no model and no GPU.
- **Its data.** Every trigram of the history (the prompt plus the accepted output) maps to its 4 most recent end
  positions in an open-addressing hash table, sized for `max_context + 4096` tokens at load factor <= 0.5 (~20 bytes
  per history token).
- **A proposal.** Checks those candidates, extends each match backwards up to 64 tokens, and keeps the longest (the
  newest on ties). A match shorter than 3 (`--suffix-draft`) proposes nothing. It proposes up to window cap - 1 = 5
  tokens.
- **It is reset at each request** and refilled with the prompt, then fed each emitted token.

### The draft policy (`include/strata/spec/draft_policy.hpp`, `src/spec/draft_policy.cpp`)

Each round, *only if* the suffix drafter proposes something *and* its first token equals the MTP's first draft, the
policy compares expected committed tokens per millisecond:

- **MTP:** the mean tokens committed by MTP windows of that size (an EMA, weight 0.05), over the measured round time
  of that size (EMA, weight 0.1). Before data: 1 + 0.7 per draft.
- **Lookup:** for each k up to the proposal, 1 + q + q² + ... + q^k over the round time of k+1. q is the acceptance
  rate of lookup drafts in one of 4 match-length buckets (<6, <12, <24, >=24), with decayed counts (0.97) and priors
  0.75 / 0.88 / 0.93 / 0.96 worth 4 observations.
- **Unmeasured sizes** are scaled from measured ones by a prior shape (1, 1.35, 1.7, 2.05, 2.45 ... per size).

The lookup window is taken when its best rate beats the MTP's by 3%. A confident lookup (q >= 0.85) also probes a
size it has measured fewer than 3 times. The policy learns over the whole process, not per request. **Measured:**
code edits +6-11% faster, other text unchanged (DETAILS.md, `bench/results/2026-09-27-spec` cited, not in the tree);
taking every lookup that proposed more than the MTP lost 2-8% on ordinary text (`draft_policy.hpp`).

## 3. Acceptance and rollback

**Acceptance** (serve loop). The window returns T picks; pick t is the model's token after row t. Draft row t+1 is
accepted while it equals pick t, from the front: a = the number of leading drafts that match. The window commits
a + 1 tokens (row 0 and the accepted drafts). It emits picks 0..a: the accepted drafts' own values, then one more
token the model chose itself (pick a), which becomes row 0 of the next window. Under sampling the picks are sampled
tokens and the comparison is the same exact match (section 6). End of turn (`<|endoftext|>`, `<|im_end|>`),
`max_new` or a `STOP` line ends the loop between windows.

**What a rejected token touched, and how each part is put back:**

| State | During the window | Rollback of the rejected rows |
| --- | --- | --- |
| QSA K/V cells (12 layers) | appended for all T positions | nothing: positional. A cell past the accepted prefix is overwritten when its position is processed again, before any query can read it |
| QSA indexer key tail (the unfinished block's raw keys, 3 x 128 floats per layer) | appended for all T positions; a snapshot is taken at the window's start | the commit restores the snapshot and re-appends the accepted keys (a rejected key can land in a slot the current block still needs) |
| QSA indexer pooled keys / block positions | written when a block completes | rewritten by the re-append in the commit |
| GDN recurrent state and conv history (36 layers, ~113 MB) | read, never written | nothing to undo: the commit replays the accepted rows (section 8) |
| n-gram (PLE) history (9 x 10240 normalized floats) and the last-two-tokens record | advanced per row on the device; a snapshot after every row | the commit copies the snapshot after row a back; the host advances its two-token record by the a+1 committed tokens |
| MTP draft layer K/V | catch-up writes all T cells, the draft chain writes speculative cells | nothing: positional, overwritten next round |
| Suffix drafter history | - | only emitted tokens are appended |
| Penalty history (host list of consumed tokens) | - | only committed tokens are appended |

**Measured.** Exact: the output equals plain greedy decoding token for token, including with deliberately wrong
drafts (paper §3.3, finding 3). The tests drive this with `--spec-oracle` (drafts from the expected continuation) and
`--spec-corrupt N` (every Nth draft wrong). Speed: not measured separately (rollback is part of the commit).

## 4. One layer inside a window

**The sequence** (`Verifier::record_window`'s `pre` and `post`, `Verifier::run`'s host loop,
`expert_pool_dispatch_multi`). On the GPU, inside the one window graph:

1. **pre(l):**
   - the hyper-connection read (first half) and the layer's mixer (GDN or QSA) for all T rows;
   - the second hyper-connection read and the router (one GEMV for the T rows, one top-10 kernel);
   - [E-6 only] the device plan;
   - **the doorbell:** the MoE input (T x 2560 floats), the ids and the weights are written to mapped memory, then a
     system-scope fence, then the sequence counter is incremented with a volatile store;
   - the shared expert (its own kernel) and the activation quantization for the grouped expert kernels.
2. **post(l):**
   - wait for flag A (the plan), then copy the plan block in;
   - the VRAM groups (the grouped expert kernel over resident slots);
   - wait for flag B (the PCIe share has landed), stage it if in kernel mode, then the PCIe groups;
   - wait for flag M (the CPU's rows), then copy only the rows the CPU computed;
   - add the GPU's rows, then combine: the weighted sum of the 10 rows plus the shared expert. That becomes the
     layer's pending write, which the next layer's first read folds in.

On the host, one pool thread per window spins on the counter for layer l (`_mm_pause`, no driver call). Then it:

1. **Plans the layer:** the distinct experts of the window's T x 10 entries, in routing order. Resident ones become
   VRAM groups. The last `pcie_frac` share of the missed ones becomes PCIe groups (at most 16, the staging slots, and
   only experts in pinned memory). The rest stay with the CPU.
2. **Publishes the plan:** a fence, then **flag A** = l+1.
3. **Starts the PCIe share** (mode 0, DMA): one async host-to-device copy per expert on the copy-engine stream, then
   a host function on that stream that raises **flag B**. In modes 1 and 2, flag B rises at once and the GPU reads
   the arena itself.
4. **Computes its share:** quantizes the T activations, groups the entries by expert (one job per distinct missed
   expert, all its tokens together), and runs the CPU pool. Rows the GPU owns are written as zeros.
5. **Raises flag M** = l+1. If the pool published no plan, the host writes an empty one and raises A and B first.

```mermaid
sequenceDiagram
  participant G as GPU (window graph)
  participant M as Mapped host memory
  participant H as Host pool thread
  participant C as Copy engine
  G->>G: hc read, mixer, hc read, router (T rows)
  G->>M: MoE input, ids, weights; fence; seq = l+1 (doorbell)
  G->>G: shared expert, quantize
  H->>M: spin until seq >= l+1
  H->>H: plan: VRAM groups, PCIe share, CPU share
  H->>M: plan block; fence; flag A = l+1
  H->>C: DMA PCIe share into staging (mode 0)
  C->>M: flag B = l+1 (host function when landed)
  G->>M: wait A >= l+1, copy plan
  G->>G: VRAM groups
  H->>H: CPU experts, in place in RAM
  G->>M: wait B >= l+1
  G->>G: PCIe groups (mode 2: copy kernel stages them first)
  H->>M: CPU rows; fence; flag M = l+1
  G->>M: wait M >= l+1, copy CPU rows
  G->>G: add, combine, next layer
```

**The flags.**

| Flag | Meaning | Written by | Waited on by |
| --- | --- | --- | --- |
| seq (the doorbell) | layer l's ids and MoE input are in host memory | the GPU: fence, then volatile increment | the host: spin, bounded at 20 s per layer |
| A | layer l's GPU plan is in the plan block | the host, after a fence | `wait_flag_ge` before the VRAM groups |
| B | the PCIe share is in staging (or readable in place) | a host function on the copy stream (DMA), or the host at once; only ever raised (compare-and-swap), since an earlier layer's host function may run late | `wait_flag_ge` before the PCIe groups |
| M | the CPU's rows are in the mapped result rows | the host, after a fence | `wait_flag_ge` before the copy and combine |

- **Values.** Each wait compares against a value fixed when the graph was captured (l+1, or the ring number with the
  split window). The flags are reset to 0 before each launch, so several rings can be outstanding.
- **The wait kernel.** One thread spinning on a volatile load, then a system-scope fence.
- **Memory ordering.** The doorbell's fence is required. Without it the host sees the ring late, and a compensating
  driver call per spin costs ~10 ms per token (`elementwise.cu`). On RDNA4 a plain store to mapped memory stays in L2
  until a sync: the host saw 0 of 100 rings. With a volatile or system-scope store, or a fence after it, it saw 100 of
  100 (`tests/hip/handoff`).

**The PCIe share.** How the share is sized, its three copy modes (the default is the in-graph copy kernel, not
DMA) and what it was measured to be worth are in
[part 2, the PCIe share](STRATA_MEMORY.md#4-the-pcie-share).

**E-6: the skip when every expert is resident** (`STRATA_VERIFY_DEVICE_PLAN=1`, off by default).

- **What it does.** After routing, a one-thread kernel checks the group's entries against the device copy of the
  residency table. If every one is resident, it writes the same plan the host would (same layout, same order, no
  PCIe groups) and sets a device skip word to the ring value.
- **The waits.** The A, B and M waits are `wait_flag_ge_or` and return when either the flag or the skip word reaches
  the value. The plan copy is skipped, and the CPU rows become zeros. The host still runs its loop for the layer,
  off the GPU's critical path.
- **Why it is off.** It needs all of a window's ~25 distinct experts in the layer to be resident, which is true for a
  minority of layers ([expert hit-rate record](../measurements/MEASUREMENTS.md#2026-10-12-expert-hit-rates)).
- **Measured.** Exact (device plan off/on 10 of 10 identical), and neutral for speed (commit `0abcd5b`). On an
  RTX 5080 alone: story 86.7 -> 84.5, code 103.3 -> 102.0 tokens/s. On 5080 + 3090: 81.5 -> 82.4 and
  113.9 -> 109.9.

**Measured for the layer as a whole.** About 0.6 ms per layer at 4K (paper figure 2). The GPU and the CPU each take
about half of a window and wait for each other at every layer (paper §5, finding 1).

## 5. Per-sequence state (`include/strata/core/session.hpp`, `src/core/session.cpp`)

**What `SessionState` holds** (device memory, one per sequence, or one per stage with a layer split):

- the residual stack R (4 streams x 2560);
- 36 GDN states (recurrence + conv history; ~113 MB together per `verify_kernels.hpp`) with one shared GDN scratch;
- 12 QSA states, each with its KV cache (paged, FP16 / INT8 / Q4_0 / K8V4), a page table, the indexer's tail, dead
  key, pooled keys and block positions, and a host copy when KV streaming is on; QSA scratch is shared by the 12;
- MoE and block scratch;
- the n-gram (PLE) run: the last two tokens (host), the 9 x 10240 normalized history (device), and its scratch;
- a pointer to the doorbell.

Sizes depend on the context and the KV format (`session_bytes`). The verify window's own buffers (section 1) and the
drafter's K/V are separate. A conversation checkpoint, which holds exactly the running (non-positional) part, is
~118 MB.

**Carved per layer range.** A layer-split stage owns layers [lo, hi) and carves state only for them. The arrays keep
global ordinal indexing with an offset (`gdn_ord0`, `qsa_ord0`). A range without a QSA layer still carves one
primary QSA state for model-level uses (the RoPE table, prompt staging, the drafter's rope). `session_bytes` is pure
arithmetic, so the split search prices a placement before allocating.

**No allocations on the token path** (Strata's rule P2.T10, `session.hpp`). Every layer's state is sized at once for
`max_cells`. The verifier's arena and mapped staging are allocated in `init`, the CPU dispatch's tables are reused
every layer, and the graphs bake fixed addresses (per-token values arrive through fixed device buffers). The reason
`session.hpp` gives: a `cudaMalloc` on the first token of a longer sequence would pass every test and still break the
rule. Not measured.

## 6. Sampling

- **The chain** (`sampler.hpp`, llama.cpp's default order): penalties -> top_k -> top_p -> min_p -> temperature ->
  pick. `top_k` keeps at most 64 candidates (0 or more than 64 means 64).
- **Greedy** (the default, and what a request without sampling keys gets): the argmax of each row is recorded *inside*
  the window graph (no extra launch or sync).
- **Sampled or penalized:** after the graph, the host launches the sampler kernel again on the same stream with the
  request's own parameters, then syncs. A captured sampler would bake its parameters and replay the same draws.
  Row t of a window at p0 draws Philox(seed, p0 + t): tied to the *position*, not to how the text was cut into
  windows, so a seed gives the same text whatever the drafts were. A rejected row's draw is discarded.
- **Penalties:** one history row per window row (`penalty_rows`): the consumed tokens, row 0's token, then drafts
  1..t, so row t counts exactly what plain decode would. They are uploaded before every window.
- **Where it runs:** on the GPU, over the full-vocabulary head logits of the window's last stage. Drafts are argmax
  over the draft head's subset, or coupled samples (section 2).
- **Per request:** greedy until `set_sampling`; the server sends the keys on the `GEN` line (section 10).
- **Rows whose picks are discarded** (a short prompt part read through windows): head sampling is switched off
  (`set_head_sampling`).

**Measured.** Since 0.1.19 the penalties apply to every row of a window. Requests with penalties are 1-11% slower
than in 0.1.18, because the drafter guesses without penalties and more drafts are rejected. Requests without penalties
are unchanged (DETAILS.md). With the default adaptive expert tier, a sampled result is not reproducible run to run;
`--adapt-every 100000` makes it reproducible from the seed (DETAILS.md).

## 7. The stall watchdog (#29, #31, #251, #267)

**The problem.** A protocol bug between the host loop and the GPU does not crash: a wait kernel spins on a flag
nobody raises, the GPU shows 100%, one core is busy, and no tokens come. On Windows, killing the process while a
spin kernel is resident leaves the GPU "lost" until a power cycle (#267).

**What Strata has** (`progress.hpp`, `generate.cpp` serve block, `verify.cpp`, `src/kernels/cpu/pool.cpp`):

- **A heartbeat.** Tokens, prompt chunks and verify windows bump a counter. Each layer stores where it is (two
  relaxed stores) and a layer tick.
- **The serve watchdog thread.** If a request's heartbeat stops for `STRATA_WATCHDOG_S` seconds (default 60; 0 = off),
  it:
  1. prints the stage;
  2. prints a stall report: every pool thread, the verifier's layer, counter and flags A/B/M, memory and page faults,
     and on Windows a minidump;
  3. **releases the GPU's waits**;
  4. aborts.

  The server then starts the engine again on the next request.
- **Bounded waits in the window.** The host gives up on a layer's ring after 20 s. It also checks every 2 ms whether
  the graph already ended without ringing (an error). The final wait for the window stays a blocking stream sync.
- **The release** (`Verifier::release_gpu_waits`):
  1. Write UINT32_MAX into flags M, A and B (past every ring value). Each spin kernel then returns; the E-6 waits
     return on their flag too.
  2. Fence, then wait up to 5 s for the window and copy streams.
  3. Mark the verifier released: it refuses every later window, because a released window computed on whatever the
     flags guarded.

  It runs for every live verifier (one per layer-split stage). The 20 s timeouts, the watchdog and the CPU pool's own
  stall aborts all call it before ending the engine.
- **A test hook.** `STRATA_TEST_VERIFY_STALL=N` withholds the last layer's flag in the N-th window.

**Measured.** The watchdog is for correctness, not speed, and its effect is not measured. Polling the final wait with
stream queries instead of a blocking sync cost IQ3_S ~3% decode (a core calling the driver beside the expert
workers). Strata reverted it (commit `0e23f57`).

## 8. The commit

**The rule.** A window is verified without touching the GDN state. The verify half of each GDN kernel reads the conv
history and the recurrent state, runs all T rows in order, and writes only per-row outputs. It also keeps each
row's inputs: the projected qkv (before the conv), the post-conv q/k/v, the gate and beta, per GDN layer and row.
Once acceptance is known, `commit(n_keep)` replays the first n_keep = a+1 rows from those stored inputs and writes the
state. So there is no snapshot of the 113 MB recurrent state, and a rejected row never touches it
(`verify_kernels.hpp`).

**What the commit graph does**, per stage, in layer order:

1. Copy the commit record from mapped memory: n_keep, n_keep-1, and the kept rows' positions (-1 for the others).
2. Per GDN layer:
   - `gdn_conv_commit`: the conv history becomes the last 3 of (history, kept rows' qkv);
   - `gdn_step_norm_multi` in commit mode: the recurrence over the kept rows, writing the state; its output goes to a
     dummy.
3. Per QSA layer:
   - restore the indexer key tail from the window's snapshot (one copy kernel);
   - re-append each kept row's key, one indexer-append kernel per window row up to the verifier's maximum (rows past
     n_keep carry position -1 and return at once).
4. The stage holding layer 1: copy the PLE history snapshot after row n_keep-1 back (one kernel, index read from
   device memory).

The host then advances its two-token PLE record by the kept tokens.

**Launches per window.** One graph launch per stage per window. Inside the whole-model graph, with M = the verifier's
maximum window (6 with setup's defaults, 4 with the suffix drafter off):

| Part | Kernels | M = 6 | M = 4 |
| --- | --- | ---: | ---: |
| commit record copy | 1 | 1 | 1 |
| GDN: conv commit + recurrence replay | 2 x 36 | 72 | 72 |
| QSA: tail restore + M re-appends | (1 + M) x 12 | 84 | 60 |
| PLE history restore | 1 | 1 | 1 |
| **Total kernels** | | **158** | **134** |

[Splash](https://github.com/incoai/splash) commits every DeltaNet layer in one dispatch.

**What is restored outside the replay:** the QSA indexer's key tail (from a per-window snapshot, then re-append) and
the n-gram/PLE history (from per-row snapshots), plus the host's last-two-tokens record. K/V cells, the drafter's K/V
and the pooled indexer rows need no restore (positional, rewritten).

**When it runs.** On one GPU the commit returns without waiting (`set_commit_async`). The next window follows on the
same stream, and the drafter reads only the window's final rows and its own K/V. Anything else that touches the
session afterwards waits on an event recorded after the commit (`wait_commit`): the end of a request, the start of
the next, checkpoints, snapshots, the prompt path. A layer split keeps a synchronous commit; so does
`STRATA_COMMIT_SYNC=1`.

**Measured.** A synchronous commit holds the host ~0.45 ms per round. The async commit: IQ2_XS 400-token answers
128.6 -> 135.8 tokens/s (+5.6%), the same tokens; an NVFP4 pack +0.8% (commit `42b4299`). The commit's own GPU time
is not reported separately.

## 9. Identical tokens

**The rule** (`include/strata/core/verify.hpp`). A window's pick for row t is what plain greedy decoding produces
after token t, bit for bit. Every kernel in the window is either the single-token kernel applied per row, or a
multi-token kernel whose per-row arithmetic is the single-token kernel's. A draft is then accepted exactly when
greedy decoding would have produced it.

**The kernels it covers, and how each keeps it:**

| Kernel | How it stays bitwise per row |
| --- | --- |
| Quantized projections, multi-column MMVQ (`native_mmvq.hpp`) | "exact" layout (the default): each column uses the one-column layout. llama.cpp's multi-column table is equal only to float rounding and is off |
| Hyper-connection read for up to 8 rows (`fused_gr.hpp`) | every variant computes the plain read's operations in its order. `fused_gr_check` tests the variants against the single-token read on each card at start (1-8 rows, random inputs) and uses the newest that agrees bit for bit |
| GDN conv, gate/beta, recurrence (`verify_kernels.hpp`) | the T-row kernels are bitwise the fused single-token ones per row |
| Router GEMV + top-10 for n rows (`native_router.hpp`, `bf16_gemv`) | each row exactly as the single call |
| QSA batched projections, norms, RoPE, gate (`STRATA_DEC_BATCH`, default on) | "row-wise identical arithmetic" to the per-token calls |
| Shared expert, multi (`shared_expert.hpp`) | every token bitwise `shared_expert` on that token |
| VRAM expert rows | per-token activation quantization with the CPU's rule and fp32 scale (Strata's R4.2h). With an fp16 scale, 80 of 80 chunks differed |
| CPU expert rows, Q2_0 (`cpu/expert.hpp`) | multi-token rows bitwise the single-token rows, same order per token |
| CPU expert rows, i-quants | **the exception**: one token uses ggml's dot product, several tokens Strata's multi-token kernels, which round differently. `STRATA_IQ_MT_MIN=1` uses the multi-token kernels from one token on (#152, opt-in). `=2` restores the old rule |

**Where identical tokens still break.** The rule holds within one placement of the experts. A GPU and the CPU round
the same expert differently, so output depends on where each expert ran. The switches a byte-identical repeat needs,
and what each turns off:

| Switch | What it turns off | Why it matters |
| --- | --- | --- |
| `STRATA_IQ_MT_MIN=1` (IQ models) | the single-token CPU path | how many rows share an expert depends on the drafts. Cost: IQ3_S decode -1..-3%, the other models the same |
| `--adapt-swaps 0` | the adaptive tier's swaps between RAM and VRAM between rounds | a swap moves an expert from CPU to GPU rounding mid-conversation and across requests |
| `--pcie-frac 0` | the PCIe share of the misses (computed on the GPU) | the first answer after a start differed from the next ones. Cost: decode speed, since every miss runs on the CPU |
| `--prompt-cache 0` | conversation checkpoints and resume | a resumed prompt reads only its tail through the decode windows instead of the batched prompt path, whose arithmetic differs |
| `--adapt-every 100000` | adaptive residency (for seeded sampling) | seed-reproducible sampled output |

**Measured** (DETAILS.md, #410: IQ3_XXS, a 3.6K-token prompt, 4 repeats). With all three server switches, 1 answer of
4 differed. Without `--pcie-frac 0`, 2 of 4 (the first differs). With the defaults, 2 of 4. The exactness of the
window itself was tested with forced wrong drafts (section 3). Against llama.cpp the mean next-token KL is 0.022,
below the 0.058 between llama.cpp's own CPU and GPU builds (paper finding 3).

## 10. The server (`serve/server.py`, the engine's `--serve` block in `generate.cpp`)

**Shape.** A Python HTTP server keeps one engine process running and talks to it over stdin/stdout, one line per
message. The engine serves one sequence at a time; the server serializes requests in front of it. A `MockEngine`
replays scripted answers so every API path is testable without a GPU.

**The engine's line protocol.**

| Direction | Line | Fields |
| --- | --- | --- |
| server -> engine | `GEN <max_new> [key=value ...] <id,id,...>` | keys: `temperature`, `top_p`, `top_k` (1..64), `min_p`, `penalty_last_n` (64 when any penalty is set), `penalty_repeat`, `penalty_freq`, `penalty_present`, `seed`, `cvec=0\|1`, tuning `pcie_frac`, `spec_min_p` (this request only); unknown keys skipped; the ids start at the first token without `=` |
| server -> engine | `GENI <max_new> [key=value ...] <embeddings file> <id,...>` | the same plus a file of image-embedding records for the `<\|image_pad\|>` runs |
| server -> engine | `STOP` | read on its own thread; checked between prompt chunks and between windows |
| server -> engine | `QUIT` | end the process (stdin is then closed) |
| engine -> server | `INFO key=value ...` | at start: context, kv, kv_resident, expert slots and MiB (all tiers, and the primary card), spec, mtp_max, lookup, vram_free_mib, cvec, arena_mib, pool_workers, pcie_frac, spec_min_p, conversation-cache settings, engine version |
| engine -> server | `READY <max_context> stop` | loaded; "stop" = honours `STOP` |
| engine -> server | `RESUME <n>` | n prompt tokens are reused from the live session, a checkpoint or a parked conversation |
| engine -> server | `PP <reached> <prompt tokens> <ms> <fresh tok/s>` | prompt progress, one per chunk (also a heartbeat) |
| engine -> server | `REUSED <n>` | the prompt is read; decode starts |
| engine -> server | `T <id>` | one emitted token |
| engine -> server | `DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> <hits> <lookups> <RAM blobs> <file blobs> <file MB>` | finish is `length`, `stop` or `cancel`; later fields were appended by version, so an older server reads a prefix |
| engine -> server | `ERR <message>` | the request failed; fatal errors also end the process |

**Endpoints** (`do_GET` / `do_POST`):

| Endpoint | What |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions, stream and not, tools, `response_format` JSON (prompted, then validated) |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages, stream and not, tools, thinking |
| `GET /v1/models`, `/models` | the loaded model (and its aliases) |
| `GET /health`, `/api/health` | liveness, context, images, loaded |
| `GET /props` | chat template, context, generation defaults, model path, engine version |
| `GET /status`, `/v1/status` | what it is doing now (busy, phase, tokens, rates); `/v1/status` in llama.cpp's names |
| `GET /slots` | one slot: `id` 0, `n_ctx`, `is_processing` (empty when unloaded) |
| `GET /metrics` | the Monitor tab: engine facts (the INFO keys), live state, the last 12 requests (or all), totals, hardware |
| `GET /mcp`, `/settings`, `/api/requests`, `/api-monitor` | MCP servers and tools, shared settings, the opt-in request monitor |
| `POST /load`, `/unload`, `/v1/load`, `/v1/unload` | load or free the engine; `/unload` and the `/v1/` pair answer 409 while a request runs or waits |

**The FIFO.** One lock (`svc.fifo`) in front of the engine. A request counts itself in `queued` and waits on the lock.
Holding it, the request:

1. starts the engine if it died or was unloaded (#27);
2. streams;
3. on an early stop, sends `STOP` and drains to that request's `DONE` before releasing the lock, so the next request
   never reads the previous one's lines.

Prompt plus `max_tokens` beyond the context is refused (400), never truncated.

**Conversation memory between requests** (engine side, positional vs running state as in section 3):

- **Checkpoints** (`--prompt-cache 6`, `--prompt-cache-every 16384`, `--prompt-cache-root 2048`, `--turn-token`): up
  to 6 copies of the running state (~118 MB each), taken at the start of each new assistant turn and every 16K prompt
  tokens. The oldest (the end of the system prompt) is kept for good. A checkpoint is used only when the prompt starts
  with exactly its tokens and pictures.
- **Short parts through the windows** (`--short-read 64`): a fresh prompt part of up to 64 tokens is read through
  verify windows instead of the batched prompt path. The batched path costs ~300 ms per run however short; a window
  costs ~16 ms.
- **Conversation parking** (`--conversation-cache-mib N --conversation-cache-slots 4`, opt-in, default 0 = off). Up to
  4 whole conversations are kept in a bounded host-RAM budget, so alternating clients (an agent's controller and
  worker) do not re-read each other's history.
  - **A snapshot holds:** the running state, its checkpoints, the used K/V pages, and the draft layer's K/V.
  - **Reuse:** by exact token and image prefix with the same steering mode. No session id is needed.
  - **Limits:** oldest evicted first, and a RAM floor of 2560 MiB (`--conversation-cache-min-free-mib`). Snapshots
    are validated before any state is applied, and none survive a restart.
  - **It does not run requests concurrently.**
  - **Rejected with `--layer-split`** before loading: the shared snapshot core supports one session GPU, while a
    split carves a separate `SessionState` per stage and card. `--prompt-cache 0` also disables parking.
  - **Measured:** not measured in the docs.
