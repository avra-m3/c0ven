# ADR-0060: the control plane takes a model's rules through `ModelSetup`

**Status:** draft, 2026-11-01; not accepted. Would supersede item 4 of [ADR-0024](0024-repository-layout.md) in
part (no model interface until a second architecture exists), for the control plane, `engine/`.
**Revisit when:** generic code needs from a model something `ModelSetup` does not give, an architecture's setup
cannot be written as its functions, or `deepseek41` needs sizes `ModelMemory` lacks: weights that differ per card, a
host tier, an SSD tier, an Engram cache term. Its change extends `ModelMemory`.

## Context

- [ADR-0024](0024-repository-layout.md), item 4, put off a model interface until a second architecture exists.
  DeepSeek-V4.1-Flash, GGUF architecture `deepseek41`, is to be added beside `qwen4exp`
  ([#28](https://github.com/avra-m3/c0ven/issues/28)).
- The engine process and `tools/plan.py` import `engine.models.<arch>.setup` by the file's `general.architecture`
  (ADR-0024, item 3). They call four of its functions: `open_config`, `draft_dir`, `checkpoint_rule`, and
  `vision_file` once it moves there. Each call went through a `cast`, so the type checker did not check the module.
- Generic code held Flash-Next's rules: the image encoder's file name (`mmproj-*.gguf`) in `engine/process.py`, and
  its working memory estimate as `PlanOptions`' default. Header reads opened the path given, which holds the metadata
  only when it is the first shard.

## Decision

1. `engine/models/__init__.py` defines `ModelSetup`, a `Protocol` with four functions: `open_config` (the config for
   `open` and its memory plan), `draft_dir` (the drafter beside the model), `vision_file` (the image encoder beside
   the model) and `checkpoint_rule` (where a prompt is checkpointed).
2. Each `engine/models/<arch>/setup.py` is a module of plain functions that matches `ModelSetup`.
   `engine.process.model_module` imports it by the architecture's name and returns it as a `ModelSetup`. A host test
   assigns each setup module to a `ModelSetup`, so basedpyright checks it.
3. Code outside `engine/models/` takes from a model only what `ModelSetup` gives and the numbers in the
   `ModelMemory` its memory plan is built from, each card's working memory among them.
4. Generic code that reads a GGUF's header opens the first shard, given any shard of a split file.

## Options not chosen

- A base class that each model's setup derives from: the modules would become classes, and a missing function
  would still be caught only when called. The `Protocol` checks the plain modules as they are.
- A registry that each architecture adds itself to: the import by the architecture's name already finds the module.
- Waiting for `deepseek41`'s setup to draw the interface from the two: the four functions are those the generic code
  already calls, so none is a guess.

## Consequences

- A new architecture adds `engine/models/<arch>/` with a `setup.py` that matches `ModelSetup`, and a host test that
  checks it. `engine/process.py` and `tools/plan.py` do not change.
- A function added to `ModelSetup` is added to every architecture's setup.
- Flash-Next's memory plan, drafter and image encoder rules are the same; they move into `engine/models/qwen4exp/`.
- The front end does not import the engine (ADR-0024, item 7). It has its own copy of the shard name rule.
