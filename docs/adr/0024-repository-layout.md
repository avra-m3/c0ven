# ADR-0024: the repository's layout

**Status:** accepted, 2026-10-14. Item 1 superseded in part by
[ADR-0025](0025-the-engine-image.md): the layout gains a `Dockerfile` at the root; and by
[ADR-0028](0028-ci-without-a-card.md): it gains `.github/workflows/`.
**Revisit when:** a second model architecture is added ([ADR-0006](0006-scope.md)).

## Context

- [ADR-0022](0022-two-processes-python-and-cpp.md) splits the engine into a Python front end, a Python control plane
  and a C++20 core behind one C ABI. Kernels are OpenCL C and share layouts with the core through headers in the C
  subset both accept ([ADR-0003](0003-kernel-language.md)).
- The kernels read the step plan, the flags and the page table in place, in host memory
  ([DESIGN.md, sections 2 and 4](../DESIGN.md#2-a-request-end-to-end)). The KV layout has one definition.
- The engine runs one model architecture, `qwen4exp` (ADR-0006). Most of the code does not depend on it: Level Zero,
  memory, signalling, timing, the CPU's expert products, the scheduler, the front end. Some does: the layer program,
  its kernels, the KV layout, the kinds of sequence state, the n-gram reader, the split rules and the memory plan's
  numbers.

## Decision

1. The layout:

   ```
   CMakeLists.txt, pyproject.toml, uv.lock, .python-version
   cmake/                   the ocloc rule, the ggml fetch
   include/engine_api.h     the C ABI
   shared/                  headers compiled as both C++20 and OpenCL C 3.0: step plan, flags, quant blocks
   core/                    C++20: api/ ze/ mem/ sync/ load/ step/ cpu/ timing/ tests/
     models/qwen4exp/       layer program, state kinds, n-gram reader, KV layout
   kernels/                 OpenCL C: include/ (the builtin wrappers), quant/ sample/ exchange/, module table
     models/qwen4exp/       DeltaNet, attention, indexer, hyper-connections, norms
   protocol.py              the line protocol between the two processes
   frontend/                the process that is started, by hand or by a proxy such as llama-swap
   engine/                  the control plane: scheduler, admission, seqstate/, memory plan, timing
     api/                   ctypes mirror (abi.py), native, mock, reference
     models/qwen4exp/       tensor table, split rules, state kinds, memory-plan numbers
   tools/                   the attach tool and other engine tools
   tests/<level>/           one folder per TESTING.md level; tests/<level>/qwen4exp/ for the model's own
   ```

2. No path carries the project's name. Python packages are top-level. The C ABI's symbols carry the prefix
   `engine_`, because `open`, `poll` and `close` clash with libc.
3. Code that depends on the model architecture lives in `models/<arch>/` under `core/`, `kernels/` and `engine/`.
   `<arch>` is the GGUF architecture name. Code outside `models/` does not name the model.
4. No model interface (base class, registry) until a second architecture exists. The shared code is then drawn
   from the two.
5. `shared/` holds only headers that compile as both C++20 and OpenCL C 3.0. A build step compiles each one both
   ways.
6. `engine/api/abi.py` mirrors `engine_api.h` by hand. A level-1 test checks every struct's size and field offsets
   against a table the core exports.
7. `frontend/` never imports `engine/`; a level-1 test checks the imports. Both use `protocol.py`.

## Options not chosen

- A top-level folder or package named after the project: one more level in every path, and nothing gained.
- Model code beside generic code (the layer program in `step/`, state kinds in `seqstate/`): separating them for a
  second model would mean going through every directory.
- A model interface now: with one example it is a guess, and it adds indirection to code the design keeps plain.
- Separate Python distributions for the front end and the engine: the import test does the same job.

## Consequences

- The README links here instead of describing the layout.
- Code outside `models/` that turns out to name the model moves into `models/` when found.
