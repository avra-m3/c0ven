# ADR-0059: a library for each CPU level

**Status:** draft, 2026-11-01; not accepted.
**Revisit when:** a level's `kl_test` or speed is measured off from `zen4`'s on the same machine; the pinned ggml
gains AVX-512 paths in the dot products of a type the engine loads; or the engine gains a CPU path outside ggml.

## Context

- The CPU computes the experts the cards do not hold, with ggml's CPU kernels
  ([ADR-0004](0004-write-or-reuse.md)). ggml was built once, with the reference build's flags: AVX-512 with VBMI,
  VNNI and BF16. The core refused to open on a CPU without one of them
  ([ADR-0057](0057-the-engine-reads-the-host-it-runs-on.md)). Intel's desktop CPUs since Alder Lake have no AVX-512.
- In the pinned ggml (`ggml/src/ggml-cpu/arch/x86/quants.c`), the dot products of the i-quants and k-quants have no
  AVX-512 path: on an AVX-512 build they run their AVX2 code. AVX-512 is used there only in a helper of the
  Q4_0/Q4_1/Q8_0-style dot products (VNNI, integer sums equal to AVX2's) and in packing nibbles for a quantiser.
- ggml builds several x86-64 levels in one tree only with `GGML_BACKEND_DL`, which needs shared libraries and picks
  the level through its backend registry. The core calls ggml's CPU functions (`ggml_get_type_traits_cpu`,
  `ggml_cpu_init`) directly and keeps ggml's symbols inside the engine's library.

## Decision

1. ggml is built once for each level in `ENGINE_CPU_LEVELS`, with ggml's own flags for the level
   (`GGML_CPU_ALL_VARIANTS`): `zen4`, `icelake`, `cascadelake`, `skylakex`, `alderlake`, `haswell`, `sandybridge`,
   `sse42` and `x64`, most features first. `zen4` is the reference build's flags; `x64` needs nothing beyond x86-64.
2. Each level links into a library of its own, `libengine-<level>.so`: the same core, the same C ABI. The build
   writes the levels and their CPU features to `cpu_levels.tsv` beside them. `libengine.so` links to the first
   level's.
3. Given `libengine.so`, the control plane loads the first level in `cpu_levels.tsv` whose features are all in
   `/proc/cpuinfo`. `ENGINE_LIBRARY` set to a `libengine-<level>.so` loads that level.
4. Each library checks its own level's features at open and refuses a CPU without one, naming them.
5. The tests and tools link the first level's ggml.

## Why

- The core and its C ABI stay as they are, and ggml's symbols stay inside each library.
- On the test machine nothing changes: it has every `zen4` feature and loads the same flags as before.
- On `haswell`, a level of AVX2 CPUs, the experts' dot products run the same AVX2 code as on `zen4`. `kl_test` on IQ3_S
  writes the same bytes on both, 107 and 109 s
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-01-cpu-levels-check)). `alderlake` was not run.

## Options not chosen

- ggml's `GGML_BACKEND_DL` with `GGML_CPU_ALL_VARIANTS`: ggml as shared libraries beside the engine's, and the core's
  direct calls to ggml's CPU functions replaced by lookups in the level's library.
- One build at `haswell` for every CPU: it drops CPUs without AVX2, and the Q8_0 dot products lose VNNI.
- ggml's other levels (`ivybridge`, `piledriver`, `cannonlake`, `cooperlake`, `sapphirerapids`): they add F16C or FMA
  without AVX2, VBMI without VNNI, BF16, or AMX, none of which the experts' dot products use. A CPU of theirs loads
  the level below.

## Consequences

- Any x86-64 CPU runs the engine. Arm and other architectures do not: the build's levels and the image are x86-64 only.
- The build compiles ggml and links the engine's library nine times; `-DENGINE_CPU_LEVELS=zen4` builds one level.
- Below `haswell`, ggml's dot products take their AVX, SSE or plain C paths. On `x64`, `kl_test` gives mean KL 0.1629
  against `zen4`'s 0.1584, inside its gate, and takes 152 s against 109 s, on a CPU whose multi-row dots still use AVX2
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-01-cpu-levels-check)). A CPU without AVX2 is slower still,
  by an amount not measured. `sandybridge` and `sse42` were not run.
- The tests check ggml's CPU code at the first level only, so a test on a CPU without AVX-512 still skips.
