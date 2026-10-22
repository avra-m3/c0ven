# ADR-0001: a new engine, written from scratch on Level Zero

**Status:** accepted, 2026-10-09.
**Revisit when:** no trigger is expected. This decision is the premise of every other ADR.

## Context

Strata is a CUDA engine: ~19,600 lines of `.cu` kernels, plus host code built around CUDA streams, graphs and mapped
memory. Porting it to SYCL would carry that structure onto hardware that works differently.

## Decision

1. **The engine is new code in this repository.** Strata's code is not imported, ported or linked.
2. **The host side drives the GPUs through Intel's Level Zero API directly** (`ze_api.h`, the loader
   `libze_loader`). No SYCL, no DPC++ runtime and no OpenCL host API, in the engine or in the probes that measure it.
3. **Strata is the reference for the design:** its algorithms, its measured results, its docs and paper.
   **llama.cpp is the reference for the model's maths,** and the competitor to beat.

## Consequences

- **Kernels reach Level Zero as SPIR-V or native Xe2 binaries** through `zeModuleCreate`. They are written in OpenCL
  C and compiled ahead of time by `ocloc` ([ADR-0003](0003-kernel-language.md)).
- **There are no ready-made GEMM libraries.** oneMKL and oneDNN have SYCL or OpenCL GPU interfaces, not Level Zero
  ones. The prompt path's matrix products are the engine's own kernels, on the XMX units.
- **There is no Level Zero CPU device.** Kernels run only on the B60s. CI can compile kernels (`ocloc` works offline)
  and run host tests, but cannot run kernels.
- **The toolchain is small:** compute-runtime, IGC, `ocloc`, the Level Zero loader and headers, and a host C++
  compiler. No oneAPI. [ADR-0002](0002-pin-the-toolchain.md) pins it.
- **What Strata's code knows is written down as specs** in [docs/spec/](../spec/): the model's forward pass, the
  quant formats, Strata's decode loop and its memory system.
- **Which parts away from the GPU come from libraries** is decided in [ADR-0004](0004-write-or-reuse.md).
- **Licences:** Strata and llama.cpp are MIT. Reading their code to write specs is fine. Code is not copied by
  default; anything copied keeps its copyright notice.
- **The architecture** is in [DESIGN.md](../DESIGN.md).
