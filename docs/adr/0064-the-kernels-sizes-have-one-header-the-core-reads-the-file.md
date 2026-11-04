# ADR-0064: the kernels' sizes have one header, and the core reads the file

**Status:** draft, 2026-11-03; not accepted.
**Revisit when:** a published `qwen4exp` file has a second value for one of these sizes, or the core needs one at
compile time.

## Context

- [ADR-0062](0062-a-models-shape-comes-from-its-file.md) item 5 leaves each kernel stage to choose how a size reaches
  the kernels.
- Every published `qwen4exp` file seen so far has one value for each size this ADR covers
  ([#103](https://github.com/avra-m3/c0ven/issues/103)). The expert count had two, and has a module for each
  ([ADR-0063](0063-a-kernel-module-for-each-expert-count.md)).
- The kernels use these sizes as `__local` and private array lengths and as loop bounds, which OpenCL C needs or
  gains from at compile time.

## Decision

1. `core/models/qwen4exp/kernel_sizes.h` holds the sizes with one value in every published file. There is one set of
   kernel modules, built for those values.
2. `layout.h` includes `kernel_sizes.h` only when compiled as OpenCL C. Model code in the core cannot see these sizes:
   using one is a compile error, and it reads `Shape` instead.
3. `shape.cpp` and the tests that drive kernels include `kernel_sizes.h`. The load compares the file with these values
   and refuses a difference.
4. A structure the core and the kernels share has a named capacity in `layout.h` in place of a moved size, so the core
   still knows its layout: `MissJob` uses `MISS_USED`.
5. A size that comes to have a second value in a published file moves to a build option, as ADR-0063 did for the expert
   count.

## Options not chosen

| Option | Why not |
| --- | --- |
| Pass each size as a kernel argument | the kernels lose their compile-time array lengths and loop bounds, on the decode path, for values no file differs in |
| A `-D` list in CMake | a second place for values `shape.cpp` must also know |

## Consequences

- The kernels' source and binaries do not change.
- A file with another value for one of these sizes still does not load. The core is ready for it; the kernels are not.
