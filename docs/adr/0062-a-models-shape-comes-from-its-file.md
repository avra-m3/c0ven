# ADR-0062: a model's shape comes from its file

**Status:** draft, 2026-11-02; not accepted.
**Revisit when:** a timed run shows that a kernel's sizes cannot come from the file without losing speed, or every
published file of the architecture has one shape for a year.

## Context

- `core/models/qwen4exp/layout.h` fixes the model's sizes at compile time. The core and the kernels include it. The
  core uses its sizes on about 640 lines and the kernels on about 570.
- The core compares none of them with the file. It reads `general.architecture`, and the n-gram table's keys. A
  tensor's shape gives its byte size and nothing else.
- GGUF files of one architecture can differ in shape. Every value in `layout.h` that describes the model has a key in
  the file's metadata ([MODEL_LAYER.md](../spec/MODEL_LAYER.md#the-numbers)), or a tensor's shape for the vocabulary.
  ISTA-DASLab's Coder file is `qwen4exp` with `expert_count` 256, and `ffn_gate_exps` of shape 2560x640x256. The engine
  would read experts up to id 511 at `offset + id * slice`, past the end of each tensor, with no error.
- Kernels are compiled ahead of time ([ADR-0003](0003-kernel-language.md)) and use these sizes as array lengths and
  work-group layouts, so a value read at load cannot replace every use. A loop bound that was a constant and becomes a
  kernel argument can cost speed, and speed is the project's aim
  ([ADR-0056](0056-the-aim-is-the-fastest-performance-on-battlemage.md)).

## Decision

1. The file is the source of a model's shape. `read_shape` (`core/models/qwen4exp/shape.h`) reads it at load.
2. Until a size no longer has a constant, the load compares the file's value with the constant and refuses a difference,
   naming each one. It also compares the expert and router tensors' shapes with the file's own sizes.
3. The constants go in stages, one group a pull request ([#103](https://github.com/avra-m3/c0ven/issues/103)): those
   only the core uses, then the expert count, then the kernels' other sizes, with `Q_EMBD` last.
4. Every few pull requests, and at least once after each stage that changes a kernel, ABBA on the test machine judges
   the change ([ADR-0016](0016-how-a-change-is-judged.md)). A slowdown stops the work until it is fixed or the approach
   changes.
5. Each stage that moves a kernel's size chooses how it arrives: a kernel argument, or a module built for each value
   ([ADR-0003](0003-kernel-language.md), item 4). The choice is recorded in a later ADR.
6. `Q_ATT_SLOTS` stays, as a kernel's tuning and not a size of the model. The `DF_*` constants stay: the DFlash 2
   drafter's files have no GGUF metadata.

## Options not chosen

| Option | Why not |
| --- | --- |
| Keep the constants and only refuse other shapes | the engine could never run a file of another shape, such as Coder |
| Read only the expert count, the one size that differs between published files seen so far | the other sizes stay fixed, and the next architecture variant repeats this work |
| Remove every constant in one pull request | about 1,200 uses at once cannot be reviewed or timed |

## Consequences

- A file of another shape stops at load with a clear error, where before it would have read the wrong bytes.
- Through the stages `layout.h` holds fewer sizes. [DESIGN.md](../DESIGN.md)'s single layout header stays until the
  kernels' stages decide how their sizes arrive.
- Structures shared with the kernels, such as `MissJob`, have fixed sizes today; their stages have to change them.
