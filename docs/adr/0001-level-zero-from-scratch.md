# ADR-0001: a new engine, written from scratch on Level Zero

**Status:** accepted, 2026-10-09, by the owner. The consequence on the design docs was updated on 2026-10-09, when
the plan was rewritten for the new engine as [PLAN.md](../PLAN.md).
**Revisit when:** no trigger is expected. This decision is the premise of every other ADR.

## Context

INTEL_SYCL_PLAN.md (now rewritten as [PLAN.md](../PLAN.md)) proposed porting Strata to SYCL behind a `strata::gpu` layer, keeping
its CUDA and HIP builds. Strata's engine is CUDA code: ~19,600 lines of `.cu` kernels, plus host code built around
CUDA streams, graphs and mapped memory. A port would carry that structure onto hardware that works differently.

## Decision

1. **The engine is new code in this repository.** Strata's code is not imported, ported or linked.
2. **The host side drives the GPUs through Intel's Level Zero API directly** (`ze_api.h`, the loader
   `libze_loader`). No SYCL, no DPC++ runtime, no OpenCL host API, in the engine or in the probes that measure for
   it.
3. **Strata stays the reference for the design:** its algorithms, its measured results, its docs and paper.
   **llama.cpp stays the reference for the model's maths,** and the competitor to beat.

## Consequences

- **Kernels reach Level Zero as SPIR-V or native Xe2 binaries** through `zeModuleCreate`. They are written in OpenCL
  C and compiled ahead of time by `ocloc` ([ADR-0003](0003-kernel-language.md)).
- **There are no ready-made GEMM libraries.** oneMKL and oneDNN have SYCL or OpenCL GPU interfaces, not Level Zero
  ones. The prompt path's matrix products are the engine's own kernels, on the XMX units.
- **There is no Level Zero CPU device.** Kernels run only on the B60s. CI can compile kernels (`ocloc` works offline)
  and run host tests, but cannot run kernels.
- **The toolchain is small:** compute-runtime, IGC, `ocloc`, the Level Zero loader and headers, and a host C++
  compiler. No oneAPI. [ADR-0002](0002-pin-the-toolchain.md) pins it.
- **What Strata's code knew must now be written down:** the model's forward pass, the quant formats, Strata's decode
  loop and memory system ([P04](../precursor/P04-spec-decoder-layer.md)-[P08](../precursor/P08-spec-strata-memory.md)).
- **Which non-GPU parts may come from libraries** (tokenizer, GGUF parsing, HTTP API, CPU expert kernels) is decided
  in ADR-0004 ([P38](../precursor/P38-adr-write-or-reuse.md)).
- **Licences:** Strata and llama.cpp are MIT. Reading their code to write specs is fine. Code is not copied by
  default; anything copied keeps its copyright notice.
- **The design docs:**
  - INTEL_SYCL_PLAN.md was rewritten for the new engine as [PLAN.md](../PLAN.md). It kept the machine, targets,
    memory, concurrency, CPU-side and risk sections. Its port sections were replaced by the engine on Level Zero and
    the engine's own stages.
  - DESIGN.md ([P45](../precursor/P45-design.md)) fixes the architecture from the measurements.
    [P46](../precursor/P46-stages-and-readiness.md) redoes PLAN.md's stages, speeds and risks with them.
  - DUAL_B60_DATAFLOW.md's rules hold, read as requirements for the new engine rather than changes to Strata.
