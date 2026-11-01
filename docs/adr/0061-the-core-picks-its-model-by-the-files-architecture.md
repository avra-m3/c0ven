# ADR-0061: the core picks its model by the file's architecture, behind an abstract `engine::Model`

**Status:** draft, 2026-11-01; not accepted. Supersedes item 4 of [ADR-0024](0024-repository-layout.md) (no model
interface until a second architecture exists).
**Revisit when:** a model needs a call the C ABI's code cannot make through `engine::Model`, or the calls through it
are measured to cost a share of a step.

## Context

- [ADR-0024](0024-repository-layout.md), item 4, put off a model interface until a second architecture exists.
  DeepSeek-V4.1-Flash, GGUF architecture `deepseek41`, is being added beside `qwen4exp`
  ([#28](https://github.com/avra-m3/c0ven/issues/28)).
- The C ABI's code, `core/api/engine_api.cpp`, held a `qwen4exp::Model` and called 40 of its methods. It also had
  Flash-Next's values written into it: the image and video tokens, the image's 16-pixel patches merged 2 x 2, the KV
  page of 4 cells (the indexer's block), and the drafts a round makes by default. The sampler took its candidate type
  from `models/qwen4exp/model.h`, and `core/models/qwen4exp` was on the core's include path, so its headers' bare names
  reached every file.
- Every GGUF file names its architecture in `general.architecture`. A split file's first shard holds it; Unsloth's
  first shard holds the metadata and no tensors.

## Decision

1. `core/api/model.h` holds the values the C ABI's code and a model pass between them (`Options`, `StepIn`,
   `Candidate`, `Image`, `LoadInfo` and the counters) and an abstract class, `engine::Model`, that declares the calls
   that code makes and no others. Each architecture's class in `core/models/<arch>/` derives from it. Calls only the
   model's own tests make stay on its class.
2. `engine::open_model` reads `general.architecture` from the file's header and builds that architecture's class. Any
   other architecture is refused by name. `core/models/open.cpp` is the one place that lists the architectures.
3. The C ABI's code asks the model for what it took from Flash-Next: the image's tokens, patch size and merge
   (`image_shape`), and the drafts a round makes by default and at most (`draft_max_default`, `draft_max_cap`). The KV
   page is 4 cells for every model (`KV_PAGE_CELLS`, `shared/step.h`); a model's layout header asserts that its
   blocks fit it.
4. Each architecture lists its own core sources and kernel modules (`core/models/<arch>/CMakeLists.txt`,
   `kernels/models/<arch>/CMakeLists.txt`) and sets its own build constants, such as qwen4exp's routing profile. No
   model folder is on the include path: a model's headers are included as `models/<arch>/...`.
5. The C ABI does not change. `engine_config.model_path` names the file, and the file names the architecture.

## Why

With the default three drafts a decode plan of one writer makes 15 calls through `engine::Model`, and 19 with the
most, seven; a prompt chunk of 4 runs that end their prompts makes 35, or 51 with seven drafts; swap rounds and the
calls a slot or a page change adds (`kvs_drop`, `set_pages`, `copy_cells`) are left out. An indirect call costs about
20 ns (*estimate*: a vtable load that hits the L2 cache, about 14 cycles, and a mispredicted branch, about 20, rounded
up to 60 cycles at an assumed 3 GHz), so 19 calls add 0.4 µs, 0.1% to 0.3% of the host work a token, 0.28 ms with one
writer and 0.12 ms with four ([record](../measurements/MEASUREMENTS.md#2026-10-25-decode-trace-today)), where a plan
yields one token or more; and 51 calls add 1 µs to a chunk of 64 tokens or more, which takes 650 ms or more
([record](../measurements/MEASUREMENTS.md#2026-10-17-prefill-timings)).

## Options not chosen

- One shared library per architecture, the C ABI's code compiled against each model's class with no virtual calls,
  and the control plane choosing the library by the file's architecture: two builds of that code and of every test
  that links it, to save 15 to 19 indirect calls a decode plan (see Why).
- Default bodies in `engine::Model` for the calls one architecture does not need (images, the draft layer): every
  model states each call, so a missing one fails to compile.
- A registry that each architecture adds itself to: one line a model in `open.cpp` is shorter while there are two.

## Consequences

- Flash-Next runs the same kernels, plans and values as before; only the C ABI's calls into it are indirect. Its
  class is `final`, so its calls to its own methods stay direct.
- A second architecture adds `core/models/<arch>/` and `kernels/models/<arch>/`, one line in `open.cpp`, and its
  Python side in `engine/models/<arch>/`, which the control plane already picks by the same key
  (`engine/process.py`).
- The interface has the calls of both drafter families and of images. A model without them returns false or 0 from
  `drafting`, `dflash` and `image_tokens_max`, and is then asked to draft nothing and given no images.
