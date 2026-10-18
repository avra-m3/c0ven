# Design: how the engine is built

The architecture the engine is written to. Where the core departs from it,
[ADR-0027](adr/0027-the-c-abi-as-the-core-was-optimised.md) lists each item, and [DEVELOPING.md](DEVELOPING.md) says
how a step runs now. Decisions live in the [ADRs](adr/README.md); this page says how they fit together. Every number
links the record it comes from, or is labelled *estimate*. Where the code lives is
[ADR-0024](adr/0024-repository-layout.md).

Records cited often: [the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs) (stand-in kernels on the
machine, for the decode step, step lists, the boundary, observability and the prompt path), [the card-to-card
record](MEASUREMENTS.md#2026-10-12-card-to-card), [the host memory record](MEASUREMENTS.md#2026-10-12-host-memory)
and [the CPU bar record](MEASUREMENTS.md#2026-10-11-cpu-bar) (ggml's CPU backend on this model's experts). The
machine is in [MEASUREMENTS.md#the-test-machine](MEASUREMENTS.md#the-test-machine).

## Contents

1. [The shape: two processes, two planes](#1-the-shape-two-processes-two-planes)
2. [A request, end to end](#2-a-request-end-to-end)
3. [The boundary: Python above, C++ below](#3-the-boundary-python-above-c-below)
4. [Sequence state: the KV cache and the rest](#4-sequence-state-the-kv-cache-and-the-rest)
5. [Two cards: every layer split across both](#5-two-cards-every-layer-split-across-both)
6. [Memory](#6-memory)
7. [Observability: a little always on, the rest on demand](#7-observability-a-little-always-on-the-rest-on-demand)
8. [What was taken from other engines, and what was left](#8-what-was-taken-from-other-engines-and-what-was-left)
9. [The decode loop and the CPU side](#9-the-decode-loop-and-the-cpu-side)
10. [Later optimisations](#10-later-optimisations)
11. [Open questions](#11-open-questions)

## 1. The shape: two processes, two planes

Decided in [ADR-0022](adr/0022-two-processes-python-and-cpp.md).

```
 clients ─► llama-swap ─► ┌─ FRONT-END PROCESS (Python) ──────────────────────────┐
                          │ HTTP: OpenAI and Anthropic APIs, streaming           │
                          │ chat template, tokenizer, detokenizer, tools, JSON   │
                          └───────────────┬───────────────────────────▲──────────┘
                            token ids in  │  line protocol over a pipe │  token ids out
                            (request id on every line)                │
                          ┌─ ENGINE PROCESS ──────────────────────────┴──────────┐
     CONTROL PLANE        │ Python: scheduler (ADR-0017), admission, sequence-   │
     (policy)             │ state bookkeeping (section 4), draft policy, the     │
                          │ memory plan, reading the timing records              │
                          ├───────────── engine_api.h, a C ABI ──────────────────┤
     DATA PLANE           │ C++ core: Level Zero, every allocation, recorded     │
     (mechanism)          │ command lists, one host thread, the CPU miss         │
                          │ workers, the n-gram reader, swaps,                   │
                          │ the timing ring                                      │
                          └───────┬───────────────────┬─────────────────┬────────┘
                         ┌────────▼───────┐  ┌────────▼───────┐ ┌───────▼────────┐
                         │ card 0: half   │◄►│ card 1: half   │ │ CPU workers:   │
                         │ of every layer │  │ of every layer │ │ expert misses  │
                         └────────────────┘  └────────────────┘ └────────────────┘
              miss jobs and step inputs in host memory; exchanges through host memory or the other card's
```

- The front-end process runs as a service of its own and keeps the model's port. llama-swap proxies to it, and wakes
  and sleeps the engine through it ([ADR-0032](adr/0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md)).
  It handles everything about text: the HTTP APIs, the chat template
  ([MODEL_IO.md, section 4](spec/MODEL_IO.md#4-the-tokenizer-and-the-chat-template)), tokenizing, detokenizing,
  streaming, tool calls and JSON output. It starts the engine process as its child and talks to it in
  token ids. If the engine dies, the front end reports the error and starts it again.
- The engine process holds both cards. It must be one process: host memory that both cards read needs one Level
  Zero context holding both cards, and memory imported into a second context serves only one card
  ([the host memory record](MEASUREMENTS.md#2026-10-12-host-memory)). So there is no process per card, unlike vLLM.
- The control plane decides what to do. It is ordinary Python: it may allocate, take locks and log.
- The data plane does it, under strict rules:
  - everything it uses is allocated before the engine reports ready;
  - no locks and no system calls on the token path; the driver calls are submitting recorded command lists,
    resetting and signalling events, and waiting for the lists to finish;
  - fixed shapes: one recorded list per batch width ([ADR-0017](adr/0017-scheduling-several-requests.md));
  - the host and card 0 hand over expert misses through flags in host memory, each with one writer
    ([ADR-0011](adr/0011-cpu-gpu-signalling.md)); the cards' exchanges and the n-gram rows are signalled by events.
- The step plan is the only thing that crosses from the control plane to the data plane (section 3).

## 2. A request, end to end

1. Front end. The front end checks the request, renders it through the chat template, tokenizes it and sends
   `GEN <request id> <settings> <token ids>`. A request whose prompt plus `max_tokens` exceeds the context is refused,
   never truncated. The other lines of the protocol follow Strata's
   ([STRATA_DECODE.md, section 10](spec/STRATA_DECODE.md#10-the-server-serveserverpy-the-engines---serve-block-in-generatecpp)),
   with a request id on each.
2. Admission. The scheduler finds a free slot, or queues the request. It looks for a shared prefix or a
   checkpoint (section 4). Then it calls `admit`, which copies the request's settings and prompt ids once into the
   slot's row in host memory that the cards read. From then on, the request is a slot index. After a `GEN` reaches an
   idle engine, the engine process waits up to 2 ms for the next line before it plans, so requests sent together read
   their prompts in one plan ([the record](MEASUREMENTS.md#2026-10-17-requests-sent-together)).
3. Each step, the scheduler builds a step plan: which slots decode, which prompt rows go in, and what changed
   since the last step (slots joined or left, pages added). `submit` writes it into one of two plan buffers in host
   memory, and the card threads submit the recorded list for that batch width. The list's first command copies the
   plan to the card, and the kernels read that copy. A word read from host memory at a launch's start makes the
   launch about 0.8 µs slower
   ([the BF16 products' record](MEASUREMENTS.md#2026-10-16-bf16-products-occupancy)).
4. The cards run the layers. They exchange partial results twice per layer and post expert misses as jobs for the
   CPU workers (section 5). At one row the partial sums go through host memory
   ([the card-to-card record](MEASUREMENTS.md#2026-10-12-card-to-card)). In a window of two rows or more, the
   mixer's out product and the MoE block's combine write them into the card's own memory, and one copy places them
   in the other card's ([ADR-0030](adr/0030-exchanges-into-the-other-card.md)). Measured: 96 exchanges of four rows
   take 0.48 ms less a step than through host memory ([the peer exchange record](MEASUREMENTS.md#2026-10-16-peer-exchange)).
5. Logits (248,320 x 4 B, about 1 MB per token) never leave the cards. The vocabulary is split, so each card sends
   only its top-k candidates. The host merges both cards' candidates and samples (`Model::step`, `engine_api.cpp`).
6. Output. Sampled ids go into an output ring. `poll` hands them to the scheduler, which updates the slot and
   sends `T <request id> <token>` to the front end, which detokenizes and streams.

The design keeps the host out from between steps, with two recorded step lists in flight per card. The core does
not do this yet: it runs one list per card per step, and the host joins every step to compute the embedding, gather
the n-gram rows, compute the misses, merge the candidates and sample
([ADR-0027](adr/0027-the-c-abi-as-the-core-was-optimised.md) item 1). Card 0 waits for the host 0.73 ms a plan at
`long-4k` ([the host glue record](MEASUREMENTS.md#2026-10-16-host-glue)).

The table gives the gap between one step's end and the next one's start, for each way of submitting steps. It is
measured on card 0 with a stand-in step of 194 kernels (48 layers of 4 kernels reading 2.5 MB each, about 1.54 ms),
from device timestamps, two runs each ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)). The
design uses two lists in flight.

| How steps are submitted | Gap, median | p99 | Host thread CPU per step |
| --- | ---: | ---: | ---: |
| All steps in one recorded list | 0.89 µs | 0.99 µs | - |
| One list per step, all submitted at once | 2.1 µs | 2.6-3.7 µs | - |
| Two lists in flight, the thread spinning | 3.07 µs | 3.4 µs | the whole step (it spins) |
| Two lists in flight, the thread checking every 50 µs and sleeping between | 3.07 µs | 3.4-27 µs | 51-59 µs |
| Three lists in flight, spinning | 3.0 µs | 3.4 µs | the whole step |
| One list in flight (the host joins every step), spinning | 3.5-4.9 µs | 16-31 µs | the whole step |
| One list in flight, checking every 50 µs | 56-58 µs | 0.10-1.17 ms | 51-58 µs |

- One host job is on every step's path: the n-gram rows. Layer 1 needs 16 rows per token from the 28.8 GB
  table on the SSD, chosen by hashing the token just sampled and the two before it
  ([MODEL_IO.md, 1.2](spec/MODEL_IO.md#12-the-n-gram-embedding-ple); [ADR-0015](adr/0015-ngram-table-on-the-ssd.md)).
  Only the host can read the SSD. For each step it hashes the tokens, serves the rows from its row cache or
  reads them with `O_DIRECT`, writes them into host memory and signals an event; the cards wait on that event just
  before layer 1. A window's rows are read together: the misses of all its rows go to the SSD in one round, and up
  to 8 rows are dequantised on the host thread. Reading them together takes layer 0's excess wait from 0.91 to
  0.19 ms a plan at `long-4k`, and to 0.25 at `long-256k` ([the host glue record](MEASUREMENTS.md#2026-10-16-host-glue)).
- A list already queued when a request stops still costs time. Its kernels check a stop word and return at once,
  but each still dispatches. The stand-in step takes 0.74-0.75 ms skipped against 1.54 ms run (194 kernels of
  1,280 work-groups each). With two lists in flight, at most one such step follows a stop.

### Where the bandwidth goes

| What moves | Size | How often | Source |
| --- | ---: | --- | --- |
| Weights read from VRAM | about 5.1 GB per token, half on each card | every token | [MEMORY_BUDGET.md](MEMORY_BUDGET.md#what-one-decode-token-reads-computed), computed |
| A 128K-token prompt as token ids | 512 KB | once per request | 4 B per id |
| One step's inputs (tokens, positions, slots) | a few KB | every step | [`shared/step.h`](../shared/step.h) |
| Exchange between the cards | 5 KB each way (2,560 fp16 values) | twice per layer | [the card-to-card record](MEASUREMENTS.md#2026-10-12-card-to-card) |
| Activations to the CPU for a miss, and the result back | about 5-10 KB each way | each layer with a miss | *estimate*: 2,560 values each way |
| Sampled token ids | bytes | every step | |

The weights are the only large data, and they never move: each weight byte is read once per step from the memory
nearest the work. Everything else is small, so its cost is waiting and synchronising.

RAM bandwidth is shared by the CPU workers, both cards reading host memory, and everything else on the host. Under
a competing RAM load the CPU's miss work is 1.6-2.2 times slower
([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)). So the front end and the control plane stay
light in RAM during decode: no large buffers per token, and no page cache for the n-gram table
([ADR-0015](adr/0015-ngram-table-on-the-ssd.md)).

## 3. The boundary: Python above, C++ below

Decided in [ADR-0022](adr/0022-two-processes-python-and-cpp.md). Python decides what to do; C++ does it.

| Python | C++ core |
| --- | --- |
| The front-end process, all of it | Level Zero: the context, queues, recorded command lists |
| The scheduler, admission, prefix sharing, the draft policy | Every allocation; loading the weights; the memory plan applied |
| Sequence-state bookkeeping: page ids, slots, checkpoints (section 4) | The card threads, the CPU miss workers (ggml, [the CPU bar record](MEASUREMENTS.md#2026-10-11-cpu-bar)) |
| Start-up options; computing the memory plan | The n-gram reader, swaps on the copy engines |
| Reading the timing records; the scheduler's cost model | The timing ring |
| Tests, tools and analysis (numpy) | Kernels are OpenCL C ([ADR-0003](adr/0003-kernel-language.md)) |

**Python never holds a device pointer, a Level Zero handle or a flag.**

### The interface

One header, `engine_api.h`, a C ABI, versioned and documented line by line. Python calls it through `ctypes`. The
core's ABI departs from this table; [ADR-0027](adr/0027-the-c-abi-as-the-core-was-optimised.md) lists each item, and
`engine/api/native.py` bridges them.

| Call | What it does |
| --- | --- |
| `open(config) -> info` | Load the model and allocate everything in the memory plan. Returns the engine's facts: slots, maximum context, page and checkpoint counts, where memory went |
| `admit(slot, settings, prompt_ids)` | Copy a request's settings and prompt ids into its slot, once |
| `release(slot)` | Free a slot |
| `submit(step_plan)` | Hand over the next step plan: a fixed-size struct of changes only. Does not wait |
| `poll(events) -> n` | Fill a buffer with events: tokens per slot, steps finished, prompt progress, errors. `wait(timeout)` is the same, releasing the GIL while it waits |
| `checkpoint(slot, id)`, `restore(slot, id)`, `drop(id)` | Copy a slot's fixed state to or from a checkpoint (section 4) |
| `read_timing(buf) -> n` | Drain timing records (section 7) |
| `close()` | Free everything |

The rules:

1. C++ never calls Python. There are no callbacks, so no C++ thread ever waits for the GIL.
2. Only plain structs and buffers cross, with fixed layouts given in the header. Every call returns an error
   code; no C++ exception crosses the boundary.
3. Calls are coarse: about one `submit` and one `poll` per step.
4. Ownership is one-way: C++ owns all memory and threads; Python owns every decision.
5. Three implementations of the same interface:
   - the engine;
   - a mock in Python, so the front end and the scheduler are tested without a card (Strata's MockEngine idea,
     [STRATA_DECODE.md, section 10](spec/STRATA_DECODE.md#10-the-server-serveserverpy-the-engines---serve-block-in-generatecpp));
   - the CPU reference ([ADR-0012](adr/0012-correctness-reference.md)), for the tests.

### What crossing costs

Measured on card 0 of the machine, Python 3.14.4 and `ctypes` against the same calls in C++, one thread pinned
to CPU 2, two runs each ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)):

| Call | C++ | Python |
| --- | ---: | ---: |
| A Level Zero call that does almost nothing (`zeKernelSetArgumentValue`) | 0.007 µs | 0.30-0.54 µs |
| Submitting a recorded 48-kernel list | 1.15-1.55 µs | 1.73-2.30 µs |
| That list's wall time, submit to synchronize | 37.4-39.6 µs | 38.6-39.3 µs |
| Writing a 256-byte step plan into host memory | - | 0.40-0.43 µs |

Two calls per step cost a few microseconds. A token at the decode target of 30 tokens/s
([ADR-0019](adr/0019-performance-target.md)) takes about 33 ms. The interface could be much finer and still not
matter; it is coarse for clarity and safety.

## 4. Sequence state: the KV cache and the rest

A sequence's state is more than its KV cache. Only 12 of the 48 layers have one
([MODEL_LAYER.md](spec/MODEL_LAYER.md#37-the-kv-layout-and-what-is-stored-per-token)):

| State, per sequence | Size | Behaviour |
| --- | ---: | --- |
| K, V and indexer keys, 12 attention layers, 8-bit ([ADR-0008](adr/0008-kv-cache.md)) | 14,688 B per token; 3.59 GiB at 262K | Grows with the context. Paged. Can be shared by prefix |
| DeltaNet state of 36 layers, and the n-gram embedding's history ([MODEL_LAYER.md, 2.7](spec/MODEL_LAYER.md#27-state-per-sequence)) | 118,038,528 B | Fixed. Overwritten by every token. Copied only at checkpoints |
| The draft layer's KV ([MEMORY_BUDGET.md](MEMORY_BUDGET.md#state-and-the-rest-computed)) | 35,651,584 B | Fixed: a ring of the last 32,768 positions |

The engine's sequence-state manager has three parts. The core does not page its KV yet: it keeps every slot's
whole-context KV in VRAM ([ADR-0027](adr/0027-the-c-abi-as-the-core-was-optimised.md) item 2).

| Part | What it does | Where |
| --- | --- | --- |
| Bookkeeping | Which pages and slots belong to which sequence; free lists; prefix reuse by page hash; admission by page count; checkpoints and parking; eviction between requests | Python, beside the scheduler. It handles only integers |
| Layout | Page size (4 cells, one indexer block, as in [Strata](spec/STRATA_MEMORY.md#5-the-kv-cache)); the 8-bit format; where each cell's K, V and indexer key sit; the page table | One header in the C subset both C++ and the kernels include ([ADR-0003](adr/0003-kernel-language.md)). The only definition of the layout |
| Access and residency | The attention kernels reading the pages the indexer picked; writing new cells; KV streaming (which pages stay in VRAM, which are read from host memory) | C++ core: its own set of KV kernels, run in the same recorded lists as attention |

Residency cannot move to the host. The indexer picks the pages each token, on the card, and a host round trip per
layer would cost more than it saves. Strata does it in its recorded graphs with a CLOCK sweep
([STRATA_MEMORY.md, section 5](spec/STRATA_MEMORY.md#5-the-kv-cache)).

- Kinds of state, one rule each: paged KV, fixed state slots with a pool of checkpoints, and the draft ring. vLLM's
  hybrid manager likewise keeps one spec per layer type ([`hybrid_kv_cache_manager.md`][v-hybrid],
  [`kv_cache_interface.py`][v-kvspec]).
- Prefix sharing follows from the model. A shared prefix can reuse its KV pages, but its DeltaNet state only
  through a checkpoint taken where the prefix ends. So later requests wait for the producer's checkpoint
  ([ADR-0017](adr/0017-scheduling-several-requests.md)).
- Rollback. Attention cells past the accepted prefix are overwritten; the DeltaNet state is restored by
  replay or snapshots ([MODEL_LAYER.md, 2.8](spec/MODEL_LAYER.md#28-what-must-be-saved-to-roll-back-rejected-draft-tokens)).
  The decode loop makes that choice (section 9).
- On two cards. Each attention layer has 2 KV heads: one per card. The DeltaNet state splits by head the same
  way. The indexer's keys are split by position between the cards (below). The logical page ids are the same on
  both cards, so Python keeps one page table and the core writes it to both.
- Across the boundary, sequence state adds only these: page-table changes per slot in the step plan (vLLM's
  `new_block_ids`, [`sched/output.py`][v-schedout]); `checkpoint`, `restore` and `drop`; and the page and checkpoint
  counts `open` reports. The kernels never see a request, and Python never sees a byte of state.

vLLM's sparse-attention offload draws the same line ([`hisparse.md`][v-hisparse], experimental). Its scheduler side
owns the logical host blocks and the residency policy. Its worker side owns the bytes, the copies and the GPU's
replacement state.

### The indexer on two cards: split by position

Each card keeps the indexer keys of half the blocks and scores its half. The two cards exchange their best 512
candidates, and both merge them to the same selection. The keys are never copied to both cards. This adds one
exchange per attention layer.

The indexer has one key head that all 4 query heads score against, and it reads every cached key each token:
1,632 B per token of context, 427,819,008 B at 262K ([MEMORY_BUDGET.md](MEMORY_BUDGET.md#per-sequence)). The table
compares three ways to place the keys, for one sequence. All its numbers are *estimates*. Experts lost are counted
with each expert split by rows (about 1.02 MB per card of a 2.05 MB expert, [MEMORY_BUDGET.md](MEMORY_BUDGET.md)).
Time assumes the indexer reads at ~400 GB/s, near the dense kernels' rates in
[the kernel spike record](MEASUREMENTS.md#2026-10-11-kernel-spike), and ~10 µs per extra exchange
([the card-to-card record](MEASUREMENTS.md#2026-10-12-card-to-card)).

| Way | Experts lost at 262K | Indexer time per token at 4K / 128K / 262K |
| --- | ---: | --- |
| Keys copied to both cards; each card scores all of them; no exchange | ~417 | 0.02 / 0.53 / 1.07 ms |
| All keys on one card; it scores them and sends its selections | ~417: the other card's spare memory cannot pair with it for experts | 0.14 / 0.65 / 1.19 ms |
| Keys split by position; each card scores half and sends its best 512 (the design) | ~209 | 0.13 / 0.39 / 0.65 ms |

Both cards hold about 19,000 of the 24,576 experts at 262K, so ~209 experts is about 1.1% of the cache. The
split by position costs the least memory at every context, and the least time above ~60K tokens. Below that, it
costs ~0.11 ms per token more than copying, about 0.3% of the target's 33 ms. Spare memory on only one card does
not speed anything up by itself: the cards run in lockstep, so the slower card sets the pace. It helps only if it
takes work off the other card or the CPU, for example whole experts that one card computes alone.

## 5. Two cards: every layer split across both

Decided in [ADR-0023](adr/0023-two-cards-tensor-split.md), which replaces [ADR-0009](adr/0009-two-cards.md).

- Each card holds half of every layer and computes half of it. An exchange combines the two halves twice
  per layer, signalled by events. At one row it goes through host memory, 10.0-10.5 µs per exchange at 1-4 tokens
  ([the card-to-card record](MEASUREMENTS.md#2026-10-12-card-to-card)); in a window it goes through the other card's
  memory ([ADR-0030](adr/0030-exchanges-into-the-other-card.md)).
- Each card holds half of every expert it caches (half its rows), so both cards work on every expert of a
  token at once and their expert caches hold the same experts.
- The CPU computes the experts that are not on the cards (the misses), each one whole, with ggml's kernels
  ([the CPU bar record](MEASUREMENTS.md#2026-10-11-cpu-bar)), in place in the host arena. The cards do not stream
  misses.
- No card is kept busy to hold its clock up. In decode both cards stay at 2,400 MHz in 99.74-100% of samples
  ([record](MEASUREMENTS.md#2026-10-16-gpu-clock-in-decode)).
- Each card owns half of everything every layer touches (weights, expert cache, KV and state heads, buffers).
  Nothing a card owns is copied to the other card. Only the exchange, twice per layer, and the expert misses'
  results cross.

The table gives a decode step of this shape, built from stand-in kernels timed to the rates in
[the kernel spike record](MEASUREMENTS.md#2026-10-11-kernel-spike), exchanges as in the card-to-card record, and the
CPU computing the misses with ggml ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)):

| One sequence, ms per token | Experts fixed (hit rate 0.931) | With swaps (0.973) |
| --- | ---: | ---: |
| One card, CPU computes the misses | 15.56-15.60 | 14.92-14.97 |
| Both cards, no misses | 8.25 | - |
| Both cards, CPU computes the misses | 10.99-11.28 | 9.76-9.86 |
| Both cards, the cards stream the misses instead | 11.19-11.28 | 10.39-10.43 |

Under a competing RAM load, the CPU path is 1.6-2.2 times slower and streaming 10-27% slower. The CPU stays the
first choice because it is simpler. Splitting misses between the two is a later optimisation (section 10).

## 6. Memory

- One Level Zero context holds both cards ([the host memory record](MEASUREMENTS.md#2026-10-12-host-memory)).
- Device memory (`zeMemAllocDevice`) holds what a card owns, allocated per layer
  ([the device memory record](MEASUREMENTS.md#2026-10-11-device-memory)).
- Host memory (`zeMemAllocHost`) holds the rest of what both cards or the CPU read: flags, step plans, the one-row
  exchanges. The core has no far KV. How much the host gives depends on how fragmented its memory is, not on how
  much is free ([MEMORY_BUDGET.md](MEMORY_BUDGET.md#ram-for-the-tensor-split)).
- The expert arena is the engine's own anonymous memory, in 1 GiB pieces from `mmap` with huge pages, and the copy
  engines read it in place ([ADR-0035](adr/0035-the-arena-in-the-engines-own-memory.md)). The runtime binds each
  copy's source range to the card when a copy list is recorded and unbinds it when the list is freed
  ([LEVEL_ZERO.md](LEVEL_ZERO.md) row 31), so the prompt path records its copy list once. xe
  moves its own host buffers under the host's reclaim, and the CPU's reads of them stall for seconds; it cannot move
  this memory ([record](MEASUREMENTS.md#2026-10-17-anon-arena-under-reclaim)). The arena counts against the engine's
  cgroup. `Options::anon_arena` false puts it in driver buffers instead, and so does a card whose
  `sharedSystemAllocCapabilities` lack concurrent read and write, with a line on stderr. The prompt path stages each
  run of adjacent experts with one region copy a matrix. The arena holds only the experts the cards do not cache. Each
  expert is in one place ([ADR-0023](adr/0023-two-cards-tensor-split.md) item 5). A swap copies the expert leaving the
  cards into the arena piece of the one entering, so every staging slot keeps its piece and the copy list stays valid
  ([record](MEASUREMENTS.md#2026-10-17-swaps-keep-the-copies)). The arena measures 20.49 GiB at 288 experts a layer
  ([record](MEASUREMENTS.md#2026-10-16-load-phases)) and 17.20 GiB at 324
  ([record](MEASUREMENTS.md#2026-10-16-more-experts-cached)).
- Shared memory (`zeMemAllocShared`) is never used: it moves pages behind the engine's back.
- Everything is allocated at load. Python computes a memory plan from the model file and the options, but the core
  does not take it yet: it fits its own expert count from `experts_per_layer` and `reserve_bytes`
  ([ADR-0027](adr/0027-the-c-abi-as-the-core-was-optimised.md) item 3). After the engine is ready, nothing is
  allocated or freed on the token path. Per-request state is an index into pools that already exist.

## 7. Observability: a little always on, the rest on demand

The engine measures how long each part of inference takes, so that it can be tuned. The core's timing records come
through `engine_read_timing`. The core does not have the shared rings or attach mode below yet
([ADR-0027](adr/0027-the-c-abi-as-the-core-was-optimised.md)).
The rule on the token path is **record, don't report**: the hot path stores small binary records; formatting,
adding up and exporting happen later, in Python. Observability adds no `printf`, lock, allocation or system call to
the token path.

| Mode | What it gives | Cost in the deployed build |
| --- | --- | --- |
| Always on | Each step's time per stage (card work, exchanges, CPU misses, waits on flags, the host's own work), summed per step; counters and histograms | About a dozen CPU records per step at ~7 ns each, and a few card timestamps at ~0.9 µs each |
| Attach | A per-kernel and per-miss timeline of both cards, the CPU workers and the exchanges; the cards' clocks and power | Nothing until a tool attaches |
| Development build | Trace points throughout, assertions, sanitizers, debug output from kernels | None: never deployed |
| Debugger | `gdb` on the C++ core | None |

- Always on, because the engine uses it. The scheduler's cost model needs measured step times
  ([ADR-0017](adr/0017-scheduling-several-requests.md) schedules prompt work by time), and `/metrics` reports
  them. Each thread writes its records into its own ring in host memory, time-stamped with the CPU's time-stamp
  counter. The cards write theirs with `zeCommandListAppendWriteGlobalTimestamp` at a few points per step.
  `read_timing` drains the rings across the boundary. A full ring counts what it dropped; nothing waits.
- The rings are sized at start, from the configuration (threads, steps held, detail level), as part of the
  memory plan, and do not change after. The region is created with `memfd_create` at that size, so it does not
  depend on the size of `/dev/shm` (64 MB in the measurement container).
- Attach mode works without a restart. The rings live in a shared-memory region. A Python tool opens it from the
  same container, sets a detail level the engine reads once per step, and streams the records:
  - at the higher level the engine runs a traced copy of each recorded list, with a timestamp after every kernel,
    and adds a record per CPU miss; detaching returns it to the plain lists, so the hot path has no branch for it;
  - a background thread samples the cards' clocks and power through Sysman (`zesFrequencyGetState`);
  - the tool writes a Perfetto timeline. The C++ side never formats anything.
- Clocks. `zeDeviceGetGlobalTimestamps` lines up the cards' timestamps with the CPU's, at start and then at
  intervals.
- The development build compiles in trace points everywhere, assertions, and ASan, UBSan and TSan on the C++
  core; kernels may write to a debug buffer or use `printf`. The deployed build compiles all of it out.
- Debuggers. In a container `gdb` can debug the engine only when it starts it, unless the container has
  `CAP_SYS_PTRACE` or `ptrace_scope` is 0. Debugging kernels on the card (Intel's GDB through Level Zero's debug
  API) needs the driver's debug switch on the host.
- What the parts cost, measured on card 0 and CPU 2, two runs each
  ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)):

  | Part | Cost |
  | --- | ---: |
  | `rdtsc` | 7.0-7.2 ns |
  | `rdtsc`, a 32-byte record into a ring, a release store of the index | 7.0-7.3 ns |
  | `zeCommandListAppendWriteGlobalTimestamp` in a recorded list (47 more per step: 1,537 to 1,579 µs, median) | ~0.9 µs each, on the card |
  | `zeDeviceGetGlobalTimestamps` (lining up the clocks) | 14.4-15.3 µs, median |
  | `zesFrequencyGetState`, one card's clock | 75-76 µs median, p99 ~400 µs |

  So the always-on records cost well under a microsecond per step on the CPU. A timestamp after every layer costs
  about 43 µs per step on each card, ~0.4% of a ~10 ms step. Lining up the clocks and Sysman are too slow for the
  token path and run on the background thread only. Idle after the run, card 0 reported 1,200 MHz.
- Not measured: the whole cost, tracing on against off, by [ADR-0016](adr/0016-how-a-change-is-judged.md)'s ABBA
  method.

llama.cpp's server keeps a coarser always-on version (pre-decode, decode, post-decode and sampling times,
[`server-context.cpp:3159`][l-timing]).

## 8. What was taken from other engines, and what was left

Read at fixed commits (2026-10-13): vLLM [`656d0d4`][v], SGLang [`e0c957e`][sg], llama.cpp [`9d7d504`][l]. This
llama.cpp is newer than the `3cf0325` the specs pin for the model's maths, because the server code has moved since.

| Idea | Where | Taken? |
| --- | --- | --- |
| The API server in its own process, talking to the engine in token ids | vLLM ([`arch_overview.md`][v-arch]); SGLang ([`engine.py:242`][sg-engine]) | Yes (section 1) |
| Detokenizing in the API process | vLLM ([`output_processor.py`][v-outproc]) | Yes |
| A separate detokenizer process | SGLang ([`engine.py:242`][sg-engine]) | No: not needed at tens of tokens per second |
| One worker process per GPU | vLLM ([`arch_overview.md`][v-arch]) | No: one context must hold both cards ([the host memory record](MEASUREMENTS.md#2026-10-12-host-memory)) |
| I/O on its own threads, so the engine loop never blocks on it | vLLM ([`core.py:1212`][v-core]) | Yes: the engine's protocol reader and writer |
| A request's data sent once, then only changes each step | vLLM ([`sched/output.py:232`][v-schedout]) | Yes: `admit`, then step plans of changes |
| A fixed table of request rows; each request keeps one row for its life | vLLM's model runner v2 ([`model_runner_v2.md`][v-mrv2], section 1) | Yes: slots |
| The CPU never writes a buffer the GPU may still be reading | the same, section 3 | Yes: two plan buffers, swapped at step boundaries |
| The GPU prepares its own next inputs, and samples | the same, sections 5-6 | No (as built): the host merges the candidates and samples (section 2) |
| Top-k from the logits first; a seeded random number generator in the kernel | the same, section 6 | Yes: also serves [ADR-0016](adr/0016-how-a-change-is-judged.md)'s reproducible runs |
| The GPU reading large host data in place | the same, section 5 (UVA) | Yes: prompt ids in `zeMemAllocHost`. The step plan is copied to each card first (section 2) |
| Recorded work for a fixed set of batch shapes | vLLM ([`cudagraph_dispatcher.py`][v-graphs]) | Yes, in [ADR-0017](adr/0017-scheduling-several-requests.md) |
| Planning the next step while the current one runs | SGLang ([`scheduler.py:1975`][sg-overlap]); vLLM's batch queue ([`core.py`][v-core]) | No (as built): the host joins every step (section 2) |
| KV in a pool allocated at start, with a free list and prefix hashing | vLLM ([`block_pool.py`][v-pool]) | Yes (section 4) |
| One spec per kind of layer state | vLLM ([`hybrid_kv_cache_manager.md`][v-hybrid]) | Yes (section 4) |
| Compute buffers reserved for the worst case at start | llama.cpp ([`llama-context.cpp:715`][l-reserve]) | Yes (section 6) |
| One process, HTTP threads posting tasks to a locked queue, one loop over the slots | llama.cpp ([`server-queue.h`][l-queue], [`main.cpp:547`][l-main]) | No: a hung card would take the API down |
| Python in the per-step loop | vLLM (much of [`model_runner_v2.md`][v-mrv2] works around its cost) | No: the loop is on the cards and in C++ |
| ZMQ between processes | vLLM, SGLang | No: a few lines per token; a pipe is enough |

## 9. The decode loop and the CPU side

### One step, one layer, on each card

A step is one window of positions for each decoding slot. With drafting
([ADR-0029](adr/0029-the-engine-process-drafts-and-swaps.md)) a slot verifies up to `draft_max` drafts in one window.
Per layer, each card:

1. Mixes the hyper-connections with its half of the `hc_*` weights' rows.
2. Runs its half of the token mixer: half the DeltaNet value heads and their state, or one of the attention
   layer's two KV heads (section 4). In an attention layer the indexer scores its half of the cached positions, and
   the cards exchange their best 512 candidates (section 4).
3. Exchange 1: the partial sums of the mixer's output projection, each card's added to the other's. Both cards
   now hold the same residual.
4. Routes the token in full, on both cards. Both pick the same 10 experts without a third exchange. Reading the
   whole router costs ~0.3 ms per token per card (0.13 GB at 455 GB/s, *estimate*); splitting it would cost a third
   exchange per layer, ~0.5 ms (48 x 10.0-10.5 µs, [the card-to-card record](MEASUREMENTS.md#2026-10-12-card-to-card)).
5. Computes its half of every cached expert and the shared expert: half the gate and up rows, and down over the
   matching half of its inputs, giving partial sums. If any expert is a miss, card 0 alone posts the job: it
   writes the activation and the missed ids into host memory and raises the job flag. Both cards know the misses
   (step 4), so card 1 posts nothing. A layer with no miss posts no job and waits for nothing; this is the skip
   ([the expert hit rates record](MEASUREMENTS.md#2026-10-12-expert-hit-rates)).
6. Card 0 waits for the CPU's answer and adds it to its partial sum; card 1 does not wait.
7. Exchange 2: the experts' partial sums, then the hyper-connection combine.

After the last layer each card computes its half of the vocabulary's logits and takes its top-k; the host merges
both cards' candidates and samples (section 2). With stand-in kernels the whole step takes 10.99-11.28 ms per token
at hit rate 0.931 (section 5).

### The CPU side

- 16 threads compute the misses: the host thread and 15 pool workers. 16 workers beat 8: 10.99-11.28 ms per
  token against 11.56-11.62 ([the design-inputs record, part 1](MEASUREMENTS.md#2026-10-13-design-inputs)).
- The design pins one worker per core of the 7945HX and puts every other thread on the second hardware threads
  (section 11). The core does not pin its threads yet.
- A job is one layer's misses. The workers compute each missed expert whole, from its place in the host arena,
  with ggml's kernels for its type ([ADR-0004](adr/0004-write-or-reuse.md)). The activation is first converted to
  the type's dot-product type, as the reference does
  ([MODEL_LAYER.md, section 6](spec/MODEL_LAYER.md#6-numerics-on-llamacpps-cpu-backend)). The workers write the
  weighted sum into host memory and raise the answer flag. One miss takes 88-95 µs median, two or more 172-212 µs
  ([the design-inputs record, part 1](MEASUREMENTS.md#2026-10-13-design-inputs)).
- The workers sleep between jobs. Workers that keep polling for 5 ms after each job make decode's steps slower on
  the shared host: p90 20.45 ms against 19.30 ([the CPU miss path's record](MEASUREMENTS.md#2026-10-15-cpu-miss-path)).
- Q2_0 has no SIMD path in ggml ([the CPU bar record](MEASUREMENTS.md#2026-10-11-cpu-bar)), so its 9 down tensors
  are the slowest misses. AVX-512 kernels written here come with batching (ADR-0004).
- RAM is shared with both cards; under a competing load the workers are 1.6-2.2 times slower (section 2).

## 10. Later optimisations

None of these is built. Each is judged by measuring it in the running engine.

- A plain-memory arena imported into the context.
- Splitting misses between the CPU and the cards. Streaming holds up better under RAM load (section 5).
- Lowering the CPU's cost per layer with misses: about 90 µs for one miss, 180-210 µs for two or more, in the
  stand-in decode step ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)).
- Finer bindings (cffi, nanobind) if the boundary ever becomes busier than a few calls per step.

## 11. Open questions

- The exchange soak test. Two early card-to-card runs gave bad checksums, cause unknown. A long soak of the exchange
  has not been run.

### Settled questions

- The arena holds only the experts the cards do not cache (section 6).
- Steps without the host: possible, with two recorded step lists in flight per card (section 2). The core does not
  run that way.
- Where the host threads run. The table gives the stand-in decode step of section 5, one sequence at hit rate
  0.931, three rounds, with three stand-in host threads (the two card threads and the Python control loop)
  ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)):

  | Placement | Token time, median, three rounds |
  | --- | --- |
  | 16 workers, one per core; nothing else | 10.65-10.89 ms |
  | 16 workers; host threads spinning on the second hardware threads of cores 12-14 | 10.73-11.05 ms |
  | 16 workers; host threads on those hardware threads, checking every 50 µs and sleeping | 10.76-10.85 ms |
  | 14 workers on cores 0-13; host threads spinning on cores 14 and 15 | 10.90-11.13 ms |
  | 14 workers; nothing else | 10.90-10.95 ms |

  The differences are within the run-to-run spread. The design puts one CPU worker on each of the 16 cores and
  its other threads on the second hardware threads, sleeping between checks. That costs nothing measurable and
  frees the cores' time (section 9). Not measured: the same with four sequences, where
  the design-inputs record's runs show the worker count mattering (8 workers against 16). It is to be measured with
  batching.
- Can the prompt reach 2,000 tokens/s at 4K? Yes. The engine's prompt path reads a 4,352-token chunk at
  2,590-2,646 tokens/s at position 0 ([the prefill timings record](MEASUREMENTS.md#2026-10-17-prefill-timings)).
  Stand-in kernels give the same answer with about 7% to spare. In that probe a 2,048-token chunk runs on both cards
  at once, each card doing its half of all 48 layers in the file's order, with the file's own expert weights in
  their own types: 512 experts x 40 tokens per layer, the experts the cards do not hold (5,550 MiB per card)
  streaming in on the copy engine, and the two exchanges per layer (10.5 MB each way, fp16) on the critical path.
  Three repetitions after a warm-up, both cards at 2,400 MHz
  ([the design-inputs record](MEASUREMENTS.md#2026-10-13-design-inputs)):

  | Per 2,048-token chunk, per card | ms |
  | --- | ---: |
  | Matrix products, DeltaNet, dequantising the dense weights, attention stand-in; one card alone | 718 |
  | The same on both cards at once | 718 |
  | ... with the experts streaming (250 ms of copying at 23 GB/s, hidden) | 722-723 |
  | ... with the exchanges, not overlapped | 801-809 |
  | ... with both | 826-829 |
  | Elementwise work (norms, the hyper-connection mixing, the convolution, gathering and scattering the experts' rows): *estimate*, ~53 GB moved per chunk at ~420 GB/s | ~125 |
  | A chunk | ~950: ~2,150 tokens/s |

  The 718 ms, measured kernel by kernel on one card, splits as follows. The routed experts take 344 ms (fused
  dequantise-and-DPAS, 11-20 TFLOPS; the kernel is bound by decoding the weights, at 60-100 GB/s of quantised
  bytes). The dense products take 161 ms (fp16 DPAS, 36-56 TFLOPS). The DeltaNet recurrence takes 130 ms (3.6 ms
  per layer, one token after another). Dequantising the dense weights to fp16 takes 81 ms (Q6_K's rate for all of
  them). Every kernel was checked against ggml's dequantisation or a CPU reference (relative error 1e-7 to 3e-4).
  The attention stand-in is two fp16 products as large as attending to 2,048 keys from every query (17 ms over the
  chunk). Not counted: the MTP layer (one more layer, about 2%).

[v]: https://github.com/vllm-project/vllm/tree/656d0d4506052a23d100c56e148c7b984828d637
[v-arch]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/docs/design/arch_overview.md
[v-mrv2]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/docs/design/model_runner_v2.md
[v-core]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/vllm/v1/engine/core.py#L1212
[v-schedout]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/vllm/v1/core/sched/output.py#L232
[v-outproc]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/vllm/v1/engine/output_processor.py
[v-graphs]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/vllm/v1/cudagraph_dispatcher.py
[v-pool]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/vllm/v1/core/block_pool.py
[v-hybrid]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/docs/design/hybrid_kv_cache_manager.md
[v-kvspec]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/vllm/v1/kv_cache_interface.py
[v-hisparse]: https://github.com/vllm-project/vllm/blob/656d0d4506052a23d100c56e148c7b984828d637/docs/design/hisparse.md
[sg]: https://github.com/sgl-project/sglang/tree/e0c957e22259cef81587b26d0a9a15ca630dcadf
[sg-engine]: https://github.com/sgl-project/sglang/blob/e0c957e22259cef81587b26d0a9a15ca630dcadf/python/sglang/srt/entrypoints/engine.py#L242
[sg-overlap]: https://github.com/sgl-project/sglang/blob/e0c957e22259cef81587b26d0a9a15ca630dcadf/python/sglang/srt/managers/scheduler.py#L1975
[l]: https://github.com/ggml-org/llama.cpp/tree/9d7d5049804be03fce6575cc9f49c058b3c48788
[l-queue]: https://github.com/ggml-org/llama.cpp/blob/9d7d5049804be03fce6575cc9f49c058b3c48788/tools/server/server-queue.h
[l-main]: https://github.com/ggml-org/llama.cpp/blob/9d7d5049804be03fce6575cc9f49c058b3c48788/tools/server/main.cpp#L547
[l-timing]: https://github.com/ggml-org/llama.cpp/blob/9d7d5049804be03fce6575cc9f49c058b3c48788/tools/server/server-context.cpp#L3159
[l-reserve]: https://github.com/ggml-org/llama.cpp/blob/9d7d5049804be03fce6575cc9f49c058b3c48788/src/llama-context.cpp#L715
