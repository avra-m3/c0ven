# ADR-0022: two processes; Python for the front end and the control plane, C++ for the core, a C ABI between

**Status:** accepted, 2026-10-13. Narrows [ADR-0005](0005-host-language-cpp20.md) to the engine core. Settles for
[ADR-0004](0004-write-or-reuse.md) where the HTTP API runs.
**Revisit when:** crossing the boundary costs a measurable share of a step, or a piece of the control plane turns out
to need per-token work.

## Context

- **The engine needs a server:** OpenAI and Anthropic APIs, streaming, tools, JSON output, the chat template
  ([STRATA_DECODE.md, section 10](../spec/STRATA_DECODE.md#10-the-server-serveserverpy-the-engines---serve-block-in-generatecpp)).
  This is a lot of code with no bearing on speed.
- **What the server and the engine exchange is small:** token ids in, token ids out, a few lines per token. A 128K
  prompt is 512 KB of ids.
- **Other engines split here.** vLLM runs its API server and its engine core in separate processes; SGLang runs its
  HTTP server and tokenizer apart from its scheduler ([DESIGN.md, section 8](../DESIGN.md#8-compared-with-other-engines)).
- **Both cards must be in one process:** host memory both cards read needs one Level Zero context holding both
  ([record](../measurements/MEASUREMENTS.md#2026-10-12-host-memory)).
- **LLMs write much of the code,** and write Python well.
- **Python can reach Level Zero cheaply.** There is no maintained binding, but `ctypes` calls `libze_loader.so`
  directly. On card 0, Python 3.14.4 added 0.3-0.5 µs per call, and a recorded list's submit took 1.73-2.30 µs
  against 1.15-1.55 µs from C++ ([record](../measurements/MEASUREMENTS.md#2026-10-13-design-inputs)).
- **What must not be Python:** threads that spin on flags and compute, such as the CPU expert workers (one per
  core, a budget of about 90 µs per layer). Python's GIL lets one thread run Python at a time.

## Decision

1. **Two processes.** The front end (HTTP, chat template, tokenizer, detokenizer, streaming) is the process
   that is started, by hand or by a proxy such as llama-swap. It starts the engine as its child and talks to it
   over a pipe in a line protocol, with a request id on every line.
2. **The front end and the engine's control plane are Python 3.14:** the scheduler, admission, sequence-state
   bookkeeping, the draft policy, the memory plan, reading timing records, tests and tools.
3. **The engine core is C++20** ([ADR-0005](0005-host-language-cpp20.md)): Level Zero, every allocation, recorded
   command lists, the card threads, the CPU workers, the n-gram reader, swaps and prefetch, the timing ring. Kernels
   stay OpenCL C ([ADR-0003](0003-kernel-language.md)).
4. **Between them, one C ABI header, `engine_api.h`,** versioned and documented line by line, called from Python
   through `ctypes`. Its calls are listed in [DESIGN.md, section 3](../DESIGN.md#3-the-c-abi-between-python-and-c).
5. **The boundary's rules:**
   - C++ never calls Python;
   - only plain structs and buffers cross, with fixed layouts; every call returns an error code;
   - about one `submit` and one `poll` per step;
   - C++ owns all memory and threads, Python owns every decision; Python never holds a device pointer, a Level Zero
     handle or a flag.
6. **Three implementations of the interface:** the engine, a mock in Python, and the CPU reference
   ([ADR-0012](0012-correctness-reference.md)).

## Why

- **Crashes stay contained.** A hung card ends the engine process; the front end reports it and starts it again.
- **The risky code stays in C++:** memory shared with the cards, atomics the cards also touch, spinning threads.
  That was ADR-0005's reason for C++.
- **The boundary is cheap:** two calls per step cost a few microseconds against a token of about 33 ms at the
  target.
- **Testing is simpler:** the scheduler and the front end run against the mock; comparisons against golden
  activations are numpy.

## Options not chosen

- **One process, as llama.cpp's server does.** Simpler, but a hung card takes the API down with it.
- **Python calling Level Zero throughout.** Measured cheap enough, but it puts device pointers, flags and atomics in
  Python, where nothing checks them.
- **All C++.** One toolchain, but the server is a lot of code that Python suits better.
- **A process per card, as vLLM does.** Two contexts cannot share host memory across both cards.
- **ZMQ or a binary protocol between the processes.** A few lines per token do not need it.

## Consequences

- [ADR-0005](0005-host-language-cpp20.md) covers the engine core only.
- The build has Python packaging, a pinned Python and the ctypes binding beside the C++ core, and the toolchain pin
  ([ADR-0002](0002-pin-the-toolchain.md)) gains Python 3.14.4.
- The front end implements the engine's protocol; ADR-0004 decides the tokenizer and the HTTP library.
