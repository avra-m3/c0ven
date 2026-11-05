# ADR-0063: a kernel module for each expert count

**Status:** draft, 2026-11-03; not accepted.
**Revisit when:** a published `qwen4exp` file has an expert count with no module, or the build time of the modules
matters.

## Context

- [ADR-0062](0062-a-models-shape-comes-from-its-file.md) item 5 leaves each kernel stage to choose how a size reaches
  the kernels: a kernel argument, or a module built for each value.
- `layer.cl`, `prompt.cl` and `route.h` use the expert count as the length of `__local` arrays (`p_moe_scan`,
  `router_group`) and of the array of keys each lane holds in registers (`RT_PER`). OpenCL C needs those lengths at
  compile time, so a kernel argument would mean a larger fixed array and a loop whose bound the compiler cannot see.
- Published files have 512 experts a layer, except ISTA-DASLab's Coder file, which has 256.
- The draft layer's checkpoint has 512 experts whatever the main file has, and runs the same kernels.
- [ADR-0003](0003-kernel-language.md) item 4 makes `-D` options per module the way to build variants.

## Decision

1. `Q_EXPERTS` is a build option of the modules that route experts. `layout.h` no longer defines it.
2. `kernels/models/qwen4exp/CMakeLists.txt` builds `qwen4exp` and `qwen4exp_prompt` for 512, and `qwen4exp_e256` and
   `qwen4exp_prompt_e256` for 256.
3. The core takes the main layers' kernels from the module of the file's count (`expert_module`), and the draft layer's
   from the 512 module.
4. The load refuses any other count, naming the counts that have kernels. A new count is one line in that file and one
   in `shape.cpp`.

## Options not chosen

| Option | Why not |
| --- | --- |
| Pass the count as a kernel argument | the 512 kernels lose their compile-time array lengths and loop bounds, on the decode path, for one file |
| Compile the module at load for the file's count | every start waits for the compiler, and the kernels are compiled ahead of time by decision ([ADR-0003](0003-kernel-language.md)) |

## Consequences

- The 512 modules' source and sizes do not change, so a file of 512 experts runs the same kernels as before.
- The build compiles two more modules for each chip.
- Each later kernel stage makes this choice again, in its own ADR. A size with many values, or several sizes at once,
  would multiply the modules.
