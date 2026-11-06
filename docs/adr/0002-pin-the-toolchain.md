# ADR-0002: pin the toolchain, without oneAPI

**Status:** accepted, 2026-10-09; versions pinned 2026-10-10. Item 3's image is the engine image of
[ADR-0025](0025-the-engine-image.md). The Python, Python-package and debugging rows were added with
[ADR-0022](0022-two-processes-python-and-cpp.md) on 2026-10-13.
**Revisit when:** never as a principle. The table changes with each upgrade.

## Context

Battlemage's compute stack changes month to month, and several 2026 reports are regressions between builds
(llama.cpp, OpenVINO nightlies). The engine runs in Docker, so the container carries its own user-mode driver stack.
The host provides only the kernel driver (`xe`) and the cards' firmware.

## Decision

1. **One pinned set of versions** for building, probing and running the engine:
   - compute-runtime, with IGC as its compiler;
   - `ocloc`;
   - the Level Zero loader and headers;
   - the host compiler and CMake;
   - the base image, by digest.
2. **No oneAPI component** ([ADR-0001](0001-level-zero-from-scratch.md)). llama.cpp's SYCL build, measured as the
   competitor, uses its own image.
3. **One container image** holds the set.
4. **Upgrades are deliberate.**
   - One pull request bumps versions and rebuilds every kernel (native binaries are tied to the IGC that built them).
   - It runs the Level Zero smoke test, then the checks of every part built so far.
   - It links a record of the results.
5. **The host's part is recorded here as minimums:** the kernel and `xe` firmware versions the set needs.

## Versions

The whole set installs from these sources, with these checksums, and the Level Zero loader sees both B60s through
it.

| Component | Version | Source | Reason |
| --- | --- | --- | --- |
| compute-runtime | **26.40.40196.8**: `libze-intel-gpu1` 26.40.40196.8-0, with `libigdgmm12` 22.10.2 from the same release. Not `intel-opencl-icd`: `ocloc` does not need it | [intel/compute-runtime 26.40.40196.8](https://github.com/intel/compute-runtime/releases/tag/26.40.40196.8), `.deb` assets; sha256 `d9c86b2bb195…` (`libze-intel-gpu1`), `053fceadb84a…` (`libigdgmm12`), from the release's `ww40.sum` | the newest release (2026-11-03). Replaces 26.35.39758.10 (2026-09-17), the first release without the multi-card host-memory mirroring of #953 and #980. Level Zero 1.18 on Battlemage, against 1.17 in 26.35 |
| IGC | **2.42.6** (`intel-igc-core-2`, `intel-igc-opencl-2` 2.42.6+23070) | [intel/intel-graphics-compiler v2.42.6](https://github.com/intel/intel-graphics-compiler/releases/tag/v2.42.6), `.deb` assets; sha256 `e20539632f2d…` (core), `7f5593cf8285…` (opencl) | the IGC compute-runtime 26.40 is built with; `libze-intel-gpu1` requires exactly 2.42.6 |
| `ocloc` | **26.40.40196.8** (`intel-ocloc` 26.40.40196.8-0) | compute-runtime 26.40.40196.8's `.deb`; sha256 `b2868d6d7958…` | ships with the runtime and uses the same IGC, so kernels built ahead of time match the runtime that loads them |
| Level Zero loader and headers (and spec version) | **loader 1.32.0** (`libze1`, `libze-dev` 1.32.0+u24.04: the release has no 26.04 build, and this one installs on 26.04), headers of **spec 1.17.24** (`ZE_API_VERSION_CURRENT` = 1.17) | [oneapi-src/level-zero v1.32.0](https://github.com/oneapi-src/level-zero/releases/tag/v1.32.0), `.deb` assets; sha256 `3c846af24f84…` (`libze1`), `4b783ed5fb93…` (`libze-dev`), computed on download (the release publishes `.sig` files, not sums) | the loader compute-runtime 26.35 is built with. 26.40 is built with 1.34.0, whose spec 1.19 headers carry features the driver does not implement; 26.40's driver does not require a loader version, so the loader stays |
| host compiler | **GCC 15.2.0** (`g++-15` 15.2.0-16ubuntu1) | Ubuntu 26.04 archive, `resolute/main` | the base image's default compiler; C++20 ([ADR-0005](0005-host-language-cpp20.md)) |
| CMake | **CMake 4.2.3** (`cmake` 4.2.3-2ubuntu2), with **Ninja 1.13.2** (`ninja-build` 1.13.2-1) | Ubuntu 26.04 archive, `resolute/main` and `resolute/universe` | the base image's own; both from the release pocket, so the pinned versions stay installable |
| base image (digest) | **Ubuntu 26.04.1 LTS**, `ubuntu@sha256:f144425ff09be612d6d9ad965196e9cdc23dae1f42110a8a11a3e9a8198759f7` | Docker Hub `ubuntu:26.04`, digest read on 2026-10-10 | the OS compute-runtime 26.40's release notes install on. Ubuntu 24.04 (`ubuntu@sha256:534baea6a22c…`) installs the same set, with GCC 14.2 and CMake 3.28 |
| minimum host kernel, `xe` firmware | **Linux 6.17.13**, **GuC 70.72.1, HuC 8.2.10, DMC 2.6** | [the test machine](../measurements/MEASUREMENTS.md#the-test-machine) | the oldest pair the set has run on. Intel validated 26.40 on kernel 7.0.0-14. On 6.18 or newer, compute-runtime #1019 applies: see the environment row |
| AOT target string | **`-device bmg-g21`** (IP 20.1.0) | [the test machine](../measurements/MEASUREMENTS.md#the-test-machine) | the IP version Level Zero reports for both cards, 0x05004000 = 20.1.0. `-device 20.1.0` builds the same binary |
| environment the engine sets | `ZE_ENABLE_PCI_ID_DEVICE_ORDER=1`; `ZES_ENABLE_SYSMAN=1`; `NEOReadDebugKeys=1` with `EnableDeferBacking=0` | Level Zero spec; compute-runtime debug keys | device 0 is always the card with the lower PCI address; Sysman reads free memory per card ([LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md), row 9); deferred backing off, so one context for both cards does not take host pages on kernels 6.18+ (compute-runtime #1019) |
| Python | **Python 3.14.4** (`python3.14` 3.14.4-1ubuntu0.2) | Ubuntu 26.04 archive | the front end and the engine's control plane ([ADR-0022](0022-two-processes-python-and-cpp.md)) |
| Python packages | **uv 0.12.22** manages them; `uv.lock` pins each package | `/usr/local/bin/uv` in the image | [ADR-0026](0026-python-tooling.md) |
| debugging tools | **gdb 17.1** (`gdb` 17.1-2ubuntu1), **strace 6.19** (`strace` 6.19+ds-0ubuntu5) | Ubuntu 26.04 archive | the development and debugger modes of [DESIGN.md, section 7](../DESIGN.md#7-timing-and-debugging). Not part of the engine |

No oneAPI component is in the set.

## Upgrades

An upgrade is the one pull request that decision 4 describes. Its first check is the Level Zero smoke test
(`core/tests/ze_smoke_test.cpp`) on both B60s, then the checks of every part built so far.

## Consequences

- A bug fixed upstream reaches the engine only through an upgrade pull request.
- Kernels are rebuilt on every IGC change, so binaries are never shipped across versions.
