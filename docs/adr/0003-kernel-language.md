# ADR-0003: kernels in OpenCL C, compiled ahead of time by ocloc

**Status:** accepted, 2026-10-09. Confirmed by the kernel spike on 2026-10-11
([record](../measurements/MEASUREMENTS.md#2026-10-11-kernel-spike)).
**Revisit when:** a feature the engine needs is unreachable from OpenCL C with the pinned stack, or a kernel keeps
missing its target under the escalation rule below. The kernel spike reached DPAS, 2D block loads, sub-group 16 and
the 256-register mode.

## Context

Level Zero loads kernels as SPIR-V or as native Xe2 binaries (`zeModuleCreate`), and does not say what language they
are written in. SYCL is excluded ([ADR-0001](0001-level-zero-from-scratch.md)). LLMs write much of the engine, so
the choice weighs speed against how likely an LLM-written kernel is to be correct.

## Decision

1. **OpenCL C 3.0, with Intel's extensions:** sub-groups, integer dot product, sub-group matrix multiply-accumulate,
   2D block IO, and the required sub-group size. OpenCL C is only the kernel language. The OpenCL host API is not
   used.
2. **Compiled ahead of time by `ocloc`** for the B60 (the AOT string is in [ADR-0002](0002-pin-the-toolchain.md)),
   and loaded with `zeModuleCreate` as native binaries. Kernels are built with `-cl-std=CL3.0`: without it `ocloc`
   compiles OpenCL C 1.2, and `dot_4x8packed_ss_int` is undeclared.
3. **One header of wrappers** for every Intel builtin used (block loads, 2D block IO, DPAS, shuffles, integer dot).
   Kernels call only the wrappers, and each wrapper is tested once.
4. **Variants are `-D` build options per compiled module:** quant type, sub-group size, tile sizes. `ocloc` 26.35
   refuses C++ for OpenCL under every `-cl-std` name.
5. **The escalation rule:** a kernel moves to inline assembly (vISA) or Intel's CM only when a measurement shows
   OpenCL C cannot reach its target. That happens one kernel at a time, with the record linked from the kernel.
6. **Every kernel** compiles offline in CI, is tested against reference values
   ([ADR-0012](0012-correctness-reference.md)), and has assembly and register spills that can be read.

## Why

- **Speed:**
  - OpenCL C goes through IGC, the same compiler backend as SYCL. Its extensions reach every Xe2 feature the design
    needs.
  - Intel's own production GPU kernels (OpenVINO, much of oneDNN) are written this way.
  - Decode is bound by memory bandwidth. The gap to peak there is data layout and access, which OpenCL C controls
    fully.
- **Correctness:**
  - It is the dialect closest to CUDA's thread and sub-group model, which LLMs know best.
  - There is a large public corpus, including Intel-specific code.
  - It is a small language.
  - Compile errors come back in seconds, without a GPU.
- **Its weak points are covered by decisions 3-5:** invented builtin names, no templates, indirect register control.

## Options not chosen

| Option | Why not |
| --- | --- |
| SPIR-V compiled by the driver at load | same source, but pays the compile cost at start-up. Kept as a fallback |
| Intel CM (`cmc`) | the highest ceiling short of assembly, but little public code and an explicit-SIMD model: more mistakes by LLMs. Available under the escalation rule |
| Inline assembly (vISA) | for single hot spots only, under the escalation rule |
| Triton (Intel's XPU backend) | pulls in PyTorch XPU; a poor fit for codebook gathers, spin-wait kernels and system-scope atomics |
| CUDA or HIP compiled to SPIR-V (chipStar) | familiar syntax, but cannot reach DPAS, block loads or register modes cleanly |
| SYCL, ESIMD | excluded by ADR-0001 |

## Consequences

- `ocloc` is part of the pinned toolchain (ADR-0002). An IGC upgrade rebuilds every kernel.
- Quant block definitions are shared between kernels and host code through headers written in the C subset that
  both OpenCL C and C++ accept.
- The prompt path's matrix product is the likeliest first candidate for the escalation rule.
