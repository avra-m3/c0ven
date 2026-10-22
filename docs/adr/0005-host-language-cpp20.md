# ADR-0005: host code in C++20

**Status:** accepted, 2026-10-09, with [ADR-0003](0003-kernel-language.md). Narrowed to the engine core by
[ADR-0022](0022-two-processes-python-and-cpp.md) on 2026-10-13: the front end and the control plane are Python.
**Revisit when:** no trigger is expected.

## Context

The host side is most of the engine: the Level Zero runtime layer, the memory system, scheduling, the CPU's share of
the experts, and what [ADR-0004](0004-write-or-reuse.md) leaves to be written here.

## Decision

1. **The engine's host code is C++20,** built with CMake.
2. **The compiler is pinned** in [ADR-0002](0002-pin-the-toolchain.md). Warnings are on, and are errors in CI.
3. **Level Zero is called directly** through its C headers (`ze_api.h`, `zes_api.h`).

## Why

- Level Zero is a C API, with C headers.
- The reference code the specs are written from (llama.cpp, Strata) is C and C++.
- LLMs write C++ well.
- Rust's compile-time checks would not cover the riskiest code: Level Zero calls, memory shared with the GPU, and
  atomics the GPU also touches. All of that would be `unsafe`, and the engine would also keep its own Level Zero
  bindings.

## Consequences

- CPU expert kernels written here are C++ with AVX-512 intrinsics.
- The headers shared with kernels use the C subset that both C++ and OpenCL C accept (ADR-0003).
