# Architecture decision records

An ADR records one decision: its context, what was decided, why, its consequences, and the measurement that would
make it worth deciding again.

- ADRs are immutable. A new ADR that conflicts with an earlier one marks it *superseded by* the new one
  ([ADR-0014](0014-how-work-is-done.md)).
- Statuses: *draft* (written, not yet accepted by the owner); *open* (the questions are written down, no option is
  chosen, and later ADRs decide them); *accepted*; *superseded by ADR-NNNN*, in whole or by item.
- A new ADR starts from [TEMPLATE.md](TEMPLATE.md).

| ADR | Decision | Status | Revisit when |
| --- | --- | --- | --- |
| [0001](0001-level-zero-from-scratch.md) | A new engine, written from scratch, drives the GPUs through Level Zero. Strata is the design reference, llama.cpp the maths reference | accepted | - |
| [0002](0002-pin-the-toolchain.md) | Pin the toolchain (compute-runtime, IGC, `ocloc`, Level Zero, host compiler), without oneAPI | accepted, versions pinned; item 3 superseded by ADR-0020, which ADR-0021 in turn supersedes | each upgrade |
| [0003](0003-kernel-language.md) | Kernels in OpenCL C, compiled ahead of time by `ocloc`; one wrapper header for Intel builtins; variants as build options; an escalation rule | accepted; confirmed by the kernel spike ([record](../RESULTS.md#2026-10-11-kernel-spike)) | a feature the engine needs is unreachable from OpenCL C |
| [0004](0004-write-or-reuse.md) | The GPU path is written here. Libraries do GGUF reading, the tokenizer, the chat template, HTTP and tests (Python); ggml's CPU backend computes the expert misses | accepted; the `gguf` row superseded in part by ADR-0026 (fetched from llama.cpp's repository) | a library blocks a measured need; the SSD reads need io_uring |
| [0005](0005-host-language-cpp20.md) | Host code in C++20 | accepted; narrowed to the engine core by ADR-0022 | - |
| [0006](0006-scope.md) | One machine, one model family, the full 262K context, several requests, behind a model-swapping proxy | accepted | the machine or the model family changes |
| [0007](0007-quant-formats.md) | Q2_0 first, then IQ2_XS and IQ3_XXS, IQ3_S last; expert kernels designed for Q2_0 and IQ3_XXS together | superseded by ADR-0018 | - |
| [0008](0008-kv-cache.md) | 8-bit KV; the context is never shortened; KV streamed to RAM | accepted | host memory cannot be pinned at the sizes KV streaming needs ([host memory record](../RESULTS.md#2026-10-12-host-memory)) |
| [0009](0009-two-cards.md) | Each card owns a range of layers; only the hand-off crosses, through host memory; no tensor split, no P2P | superseded by ADR-0023 | - |
| [0010](0010-several-requests.md) | Pipeline two requests across the cards, then batch; a sequence index in every kernel from the start | accepted; item 1 superseded by ADR-0023; items 5 and 6 made precise by ADR-0017 | the ADR's risks 5 (the cards balance badly) or 6 (batching gains little) happen |
| [0011](0011-cpu-gpu-signalling.md) | Flags in host memory and spinning waits, with a skip; events as the fallback. The skip is rare unless experts are prefetched | accepted, context corrected; confirmed by the doorbell probe ([record](../RESULTS.md#2026-10-11-doorbell)) | spinning on host memory proves unreliable or slower than events; or the skip stays rare with prefetching ([hit rates](../RESULTS.md#2026-10-12-expert-hit-rates), [prefetch](../RESULTS.md#2026-10-14-expert-prefetch)) and an event wait costs no more than a spin |
| [0012](0012-correctness-reference.md) | llama.cpp's CPU backend at a pinned commit is the correctness reference | accepted | llama.cpp is found wrong for this model |
| [0013](0013-performance-floor.md) | llama.cpp on this machine is the floor, with a stop rule | superseded by ADR-0019 | - |
| [0014](0014-how-work-is-done.md) | Work is planned in issues; ADRs are immutable; a README for people; docs not in comments; measurements committed, measurement code not | accepted, rewritten by the owner; item 5 amended: the records are kept privately | a rule costs more than it saves |
| [0015](0015-ngram-table-on-the-ssd.md) | The n-gram table stays on the SSD: `O_DIRECT` reads, a row cache, the device waits before layer 1, decode reads first | accepted; item 6 superseded by ADR-0018 | the SSD is too slow or uneven; or the RAM grows |
| [0016](0016-how-a-change-is-judged.md) | A token's arithmetic does not depend on the batch; switches make runs reproducible; a change is judged by accepted tokens and wall time per window, ABBA | accepted | - |
| [0017](0017-scheduling-several-requests.md) | Prompt work in time slices against decode; short prompts packed; one producer per shared prefix; one recorded shape per batch width; Strata's checkpoints | accepted | measuring several sequences at once shows that a count of chunks serves decode as well as a time slice, or that packing does not save the expert streaming it should |
| [0018](0018-one-file-iq3s.md) | One model file, GSQ-RCO IQ3_S; expert kernels for every type its experts use (IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL, Q2_0) | accepted | the disk gains room; the RAM is short for IQ3_S ([memory budget](../MEMORY_BUDGET.md), [host memory record](../RESULTS.md#2026-10-12-host-memory)); the i-quant kernels stay far below the dense formats |
| [0019](0019-performance-target.md) | The owner's target: 30 tokens/s decode for one request without drafting, 2,000 tokens/s prompt, IQ3_S on both B60s; a stop rule once the fast decode loop is built | accepted | the owner changes it, or the forecast built on the [design-inputs record](../RESULTS.md#2026-10-13-design-inputs) shows it out of reach |
| [0020](0020-the-sandbox.md) | Build and probe in the existing sandbox; its system packages listed in advance for the owner to add | superseded by ADR-0021 | - |
| [0021](0021-the-measurement-container.md) | Build and probe in a measurement container of our own on the machine; agents never change the host without the owner's say-so | accepted, confirmed when the container was set up | the container cannot see both cards or hold an unlimited memory lock |
| [0022](0022-two-processes-python-and-cpp.md) | Two processes: Python for the front end and the control plane, C++20 for the engine core, one C ABI between them | accepted | crossing the boundary costs a measurable share of a step |
| [0023](0023-two-cards-tensor-split.md) | Every layer split across both cards, exchanges through host memory; the CPU computes the expert misses | accepted; item 2's "no peer-to-peer" superseded by ADR-0030; item 6 superseded in part by ADR-0035 (the arena) | the running engine's timing shows misses or exchanges costing more than the alternatives |
| [0024](0024-repository-layout.md) | The repository's layout: core, kernels, front end and control plane apart; model-specific code in `models/<arch>/`; no project name in paths | accepted; item 1 superseded in part by ADR-0025 (a root `Dockerfile`), ADR-0028 (`.github/workflows/`) and ADR-0036 (`private/`) | a second model architecture is added |
| [0025](0025-the-engine-image.md) | One image, defined by a `Dockerfile` at the root, builds, tests and runs the engine; ADR-0002's versions, a non-root user; a `serve` target built `FROM` it with the engine inside | accepted, amended by the owner to add the `serve` target | the engine's run needs what a build image should not carry (the CI trigger is met by ADR-0028) |
| [0026](0026-python-tooling.md) | Python: uv pins everything, ruff formats and lints, basedpyright checks types in strict mode; `gguf` from llama.cpp's repository | accepted | a checker's false positives cost more than its findings save |
| [0027](0027-the-c-abi-as-the-core-was-optimised.md) | Fourteen places where `engine_api.h` version 2 departs from DESIGN.md's interface, each bridged for now by the native backend's adapter | open; item 14 settled in part by ADR-0029 (drafting and swaps); items 2 and 6 by ADR-0033 if accepted | each item is decided, or the ABI changes again |
| [0028](0028-ci-without-a-card.md) | CI on GitHub's runners: the image and the engine built, every kernel compiled, the Python checks and level 1 run, the GPU tests checked to skip without a card; the image pushed to GHCR from `main` | accepted | the owner sets up a runner on the machine, or a level-1 test meets a runner without AVX-512 |
| [0029](0029-the-engine-process-drafts-and-swaps.md) | The engine process drafts with the MTP layer found in `mtp/` beside the model file, and asks for adaptive swaps unless `--deterministic` is given | accepted | a request needs identical tokens without `--deterministic`, or drafting or swaps measure slower on a workload served |
| [0030](0030-exchanges-into-the-other-card.md) | A card may write what it sends the other card straight into that card's memory; the signal stays an event or a host flag; what the CPU reads stays in host memory | accepted | a driver or kernel update changes peer access, or the exchanges move to flags the cards spin on |
| [0031](0031-images-through-the-prompt-path.md) | Images in scope: the BF16 encoder streamed from host memory through the prompt path's buffers between chunks, an image split by patches across the cards, M-RoPE positions, images read from the request only | draft, not accepted | the encoder's time at one megapixel is well above the estimate, the staging slots are too small, or pinning its 0.90 GB fails |
| [0032](0032-sleep-the-front-end-stays-the-engine-keeps-the-arena.md) | Sleep: the front end stays and keeps the port; the engine child gives back the cards and keeps the arena; admin routes, a lease that fails closed, and `O_DIRECT` loads | accepted, provisional (a wake beside an image model); numbers updated 2026-10-17 | a wake swaps the host out or runs far over its estimate; an import both cards read; the owner wants the arena's RAM back while asleep |
| [0033](0033-paged-kv-kept-by-python.md) | The KV cache in pages of 4 cells; Python keeps the page table and sends its changes in each plan; on two cards a logical page keeps its parity; checkpoints hold only the fixed state | draft, not accepted | the table measures as costing decode or prompt time, or KV streaming needs pages it cannot name |
| [0034](0034-video-as-pairs-of-frames.md) | Video in scope: a video as pairs of frames, each encoded as an image of two frames with its own block of tokens; frames sampled and sized by the model's processor with the per-frame cap; transformers v4.57.0's prompt; PyAV in the front end | draft, not accepted | a video's pairs are too slow to encode between prompt chunks, or the trained prompt format differs from v4.57.0's |
| [0035](0035-the-arena-in-the-engines-own-memory.md) | The expert arena is the engine's own anonymous memory, read in place by the copy engines through a binding of each copy's source; the prompt path stages adjacent experts with one region copy a matrix, from a copy list recorded once | accepted | the runtime's copies from host memory it did not allocate change with a driver or kernel; xe fails in its userptr path; the arena is swapped out where the cgroup has swap |
| [0036](0036-a-private-repository-and-its-public-copy.md) | The code in a private repository, with this one its public copy | superseded: this repository holds the code | - |


ADRs 0002, 0003 and 0004 keep the numbers the plans gave them, so the numbers are not in the order the decisions
were made.

Older ADRs name files that are not in this repository:

| File an ADR links | What it was |
| --- | --- |
| PLAN.md, first named INTEL_SYCL_PLAN.md | the plan before the engine |
| `docs/precursor/P01` to `P51` | the pre-engine plans |
| DUAL_B60_DATAFLOW.md | the two-card design for the layer split |
| EXPERT_PREFETCH.md | the prefetch design, tried and not adopted |
| INTEL_BATTLEMAGE.md | a survey of Intel's stack, written for SYCL |
| SLEEP.md | the sleep design before it was built, now ADR-0032 and the README's [Sharing the cards](../../README.md#sharing-the-cards-with-other-models) |
