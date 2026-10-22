# ADR-0004: what the engine writes, and what it takes from libraries

**Status:** accepted, 2026-10-13.
**Revisit when:** a library chosen here blocks a measured need (speed, a missing model feature, a licence change), or
the SSD reads need io_uring.

## Context

- [ADR-0001](0001-level-zero-from-scratch.md): the engine is new code on Level Zero; Strata's code is not used.
  It does not say every part must be written here.
- [ADR-0022](0022-two-processes-python-and-cpp.md) puts the front end and the control plane in Python 3.14 and the
  engine core in C++20, with one C ABI between. Most of the parts away from the GPU sit in Python, where good
  libraries exist.
- **The CPU's expert products:** ggml's CPU backend runs IQ2_S, IQ3_XXS and IQ3_S at 5.0-5.7 GB/s per core and
  40-44 GB/s on 16 cores; IQ4_XS and IQ4_NL reach a ~50 GB/s memory ceiling on 8 cores; Q2_0 has no x86 SIMD path
  and runs 2.3 GB/s per core ([record](../measurements/MEASUREMENTS.md#2026-10-11-cpu-bar)). In the decode step it
  answers a layer's one miss in 88-95 µs, median
  ([record](../measurements/MEASUREMENTS.md#2026-10-13-design-inputs)).
- **The n-gram reads:** io_uring in the container is not measured.
- The toolchain is pinned by [ADR-0002](0002-pin-the-toolchain.md); Python packages are pinned in the project's
  `uv.lock` ([ADR-0026](0026-python-tooling.md)).

## Decision

1. **The GPU path is written here, all of it:** the Level Zero runtime layer, every kernel, the memory system (the
   arena, the expert caches, KV and state), recorded step lists, exchanges, swaps and prefetching, and scheduling. No
   library sits on it.
2. **Each part, decided** (versions as on PyPI on 2026-10-13; the lock file pins them exactly):

   | Part | Decision | Library and version | Licence | Why |
   | --- | --- | --- | --- | --- |
   | Reading the GGUF file | **library**, in Python: the control plane reads the header, the metadata and the tensor table, and passes offsets and types to the core in `open` | `gguf` 0.19.0 (llama.cpp's Python package) | MIT | The format is llama.cpp's; its own reader follows it. The core never parses the file, it reads byte ranges |
   | Tokenizer | **library**, in the front end | Hugging Face `tokenizers` 0.23.3, with the model's `tokenizer.json` | Apache-2.0 | The model ships a Hugging Face tokenizer; the reference tokens ([REFERENCE_OUTPUTS.md](../measurements/REFERENCE_OUTPUTS.md)) check it end to end |
   | Chat template | **library**, in the front end | `jinja2` 3.1.6, with the model's `chat_template.jinja` | BSD-3-Clause | The template is Jinja; rendering it by hand would drift from the model's |
   | HTTP API (OpenAI- and Anthropic-compatible) | **library**, in the front end | `starlette` 1.8.0 on `uvicorn` 0.54.0 | BSD-3-Clause, BSD-3-Clause | Small, asynchronous, streams server-sent events; no framework beyond what the two APIs need |
   | Sampling | **written**, on the cards | - | - | Logits never leave the cards ([DESIGN.md, section 2](../DESIGN.md#2-a-request-end-to-end)) |
   | CPU expert products | **library**, linked into the core: ggml's CPU backend (`libggml-cpu`, `libggml-base`) | llama.cpp `3cf03257f219afbe7334045ff7c6a06ac68c627d`, the commit [ADR-0012](0012-correctness-reference.md) pins | MIT | The rates above and the decode step's 88-95 µs per miss meet the budget. Written AVX-512 kernels (Strata's are 2.0-2.4 times faster at three tokens) are a later optimisation for batching |
   | The n-gram table's reads | **written**: a pool of threads doing `pread` on an `O_DIRECT` file, as [ADR-0015](0015-ngram-table-on-the-ssd.md) says | - | - | No dependency, and it is what Strata measured |
   | Tests | **library**: `pytest` for every test, including the core's through its C ABI; the core's own unit tests are plain executables registered with `ctest` | `pytest` 9.1.1 | MIT | One runner for the levels in [TESTING.md](../TESTING.md); no C++ test framework is needed |
   | Comparing tensors in tests and tools | **library** | `numpy` 2.5.4 | BSD-3-Clause | Golden activations and KL are arrays |
   | Command line, logging, JSON | **Python's standard library** (`argparse`, `logging`, `json`); the core takes structs across the C ABI and has none of these | - | PSF | Nothing to add |
   | Profiling the Python side | **tool**, not linked | `py-spy` 0.4.2 | MIT | The attach mode of [DESIGN.md, section 7](../DESIGN.md#7-timing-and-debugging) |

3. **Every library is pinned and fetched at a fixed version:** Python packages in `uv.lock`; ggml built from the
   pinned llama.cpp commit in the image. Nothing is vendored into the repository.
4. **No Strata code,** in any part.

## Options not chosen

- **A GGUF reader written here.** Small, but a second copy of llama.cpp's format rules; the core only needs offsets.
- **llama.cpp's C++ vocabulary code in the core.** The tokenizer belongs to the front end (ADR-0022), and the
  model ships a Hugging Face tokenizer.
- **FastAPI.** Adds request models and validation layers the two APIs do not need; Starlette is what it sits on.
- **The HTTP server in C++,** as llama.cpp's is. ADR-0022 puts it in Python.
- **Written AVX-512 expert kernels first.** Faster at several tokens, but ggml already meets the one-sequence
  budget; they come with batching.
- **io_uring now.** Not measured in the container; the thread pool is enough until it is.

## Consequences

- The core links `libze_loader` and ggml's CPU libraries, and nothing else.
- [TESTING.md](../TESTING.md) uses pytest throughout.
- ggml's speed limits the CPU path for Q2_0 (no SIMD path, 2.3 GB/s per core): 9 of the 48 down tensors are Q2_0.
