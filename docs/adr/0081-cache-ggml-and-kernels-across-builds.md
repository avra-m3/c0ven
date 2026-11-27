# ADR-0081: cache ggml and kernel builds across CMake and Docker layers

**Status:** draft, 2026-11-27. Extends [ADR-0003](0003-kernel-language.md), [ADR-0025](0025-the-engine-image.md),
[ADR-0057](0057-the-engine-reads-the-host-it-runs-on.md) and [ADR-0059](0059-a-library-for-each-cpu-level.md).
**Revisit when:** the pinned `llama.cpp` commit changes often, or `ocloc -spirv_input` stops emitting the same device
instructions as direct `.cl` compilation.

## Context

- A clean build or CI `serve` image build compiles nine static copies of `ggml` from the pinned `/opt/llama.cpp`
  commit ([cmake/ggml.cmake](../../cmake/ggml.cmake)). `ExternalProject_Add` launches all nine sub-builds in parallel
  without a Ninja job pool, multiplying `cc1plus` processes.
- Each kernel module runs the full OpenCL C frontend once per chip in `ENGINE_AOT_DEVICES` even for chips of the same
  generation (`bmg-g21` and `bmg-g31`), duplicating header parsing across `253` modules
  ([cmake/kernels.cmake](../../cmake/kernels.cmake)).
- The `serve` target in [`Dockerfile`](../../Dockerfile) copies `CMakeLists.txt` and [`VERSION`](../../VERSION) before
  building kernels and running `uv sync`, so every PR's `VERSION` bump ([ADR-0042](0042-version-numbers.md)) invalidates
  all cached kernel and Python layers.

## Decision

1. Prebuild the nine pinned `ggml` CPU level static archives under `/opt/ggml/<level>/lib` in the `engine` stage of
   [`Dockerfile`](../../Dockerfile). When `ENGINE_GGML_PREBUILT_DIR` holds those archives and `ENGINE_GGML_DIR` is
   `/opt/llama.cpp`, [`cmake/ggml.cmake`](../../cmake/ggml.cmake) links them directly. Source builds (when
   `ENGINE_GGML_DIR` differs or `ENGINE_GGML_PREBUILT_DIR` is empty) run `-G Ninja` sub-builds serialized through a
   Ninja `JOB_POOL` (`ggml=1`).
2. Compile each kernel module's `.cl` source to SPIR-V once per IP major via `ocloc compile -spv_only`, keeping the `.spv`
   in `ENGINE_KERNEL_SPIRV_DIR` (`<ip>/<name>.spv`), then compile each chip's `.bin` via `ocloc compile -spirv_input`
   with that module's `-cl-*` options ([cmake/kernels.cmake](../../cmake/kernels.cmake)).
3. Add `ENGINE_PREBUILT_KERNELS` to reuse precompiled kernel binaries and SPIR-V already present in `ENGINE_KERNEL_DIR`.
4. Order the `serve` stage of [`Dockerfile`](../../Dockerfile) into cached layers: build kernels from kernel sources and
   size headers first, install Python packages from `pyproject.toml`, `uv.lock` and `.python-version`, and copy
   `VERSION`, `CMakeLists.txt`, core, tools and Python application code last before configuring with
   `-DENGINE_PREBUILT_KERNELS=ON`.

## Options not chosen

- Building a separate Docker stage outside `engine` and `serve`: [ADR-0025](0025-the-engine-image.md) keeps two targets
  in [`Dockerfile`](../../Dockerfile) so toolchain pins cannot drift. Keeping the prebuilt archives inside `engine` and
  layer ordering inside `serve` stays within those two targets.
- Keeping `ExternalProject_Add` without `JOB_POOL`: `ExternalProject_Add` ignores Ninja job pools in CMake 4.2.3, so
  clean fallback builds would still oversubscribe memory unless throttled globally with `-j`.

## Consequences

- Builds inside the `engine` image skip compiling the nine `ggml` trees and parse kernel headers once per chip family.
- `serve` image builds on PRs that do not touch OpenCL headers, kernel sources, model size headers or `uv.lock` reuse the
  compiled kernels and `.venv` layers from BuildKit cache across `VERSION`, C++ core and Python edits.
