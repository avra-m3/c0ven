# ADR-0026: the Python code is formatted and linted by ruff and type-checked by basedpyright in strict mode

**Status:** accepted, 2026-10-14. Supersedes the `gguf` row
of [ADR-0004](0004-write-or-reuse.md)'s table in part: the same version, fetched from llama.cpp's repository.
**Revisit when:** a checker's false positives cost more time than its findings save, or Python's own typing tools
change what is usual.

## Context

- [ADR-0022](0022-two-processes-python-and-cpp.md) puts the front end and the engine's control plane in Python 3.14.
  [ADR-0002](0002-pin-the-toolchain.md) pins Python 3.14.4 and uv 0.12.22.
- The control plane mirrors the C ABI in ctypes and plans pages, checkpoints and prompt spans for the core. A wrong
  type there reaches the cards as a wrong page or a wrong offset.
- People and agents both write the code; the checks have to run the same way for both, offline, from the lock file.
- PyPI's `gguf` 0.19.0 does not know tensor type 42 (Q2_0), which the model's file uses
  ([MODEL_FILES.md](../MODEL_FILES.md)). llama.cpp's repository at the reference commit `3cf0325` has it, with the
  same version number.

## Decision

1. uv manages Python and the packages: `required-version` pins uv in `pyproject.toml`, `.python-version` pins
   Python, and `uv.lock` pins every package.
2. ruff formats and lints all Python code, with the rules set in `pyproject.toml`.
3. basedpyright checks types in strict mode over the code and the tests. Only `reportMissingTypeStubs` is off. A
   line that breaks a rule names it in `# pyright: ignore[rule]`.
4. A change passes `ruff format`, `ruff check`, `basedpyright` and `pytest` before it is committed.
5. `gguf` 0.19.0 comes from llama.cpp's repository at commit `3cf03257f219afbe7334045ff7c6a06ac68c627d`,
   `gguf-py`, pinned in `uv.lock`.

## Why

Strict mode refuses values whose type it cannot know, so every function's parameters and returns are annotated and
a library without types is typed where its results are read. basedpyright is pyright, the checker editors already
run, with stricter defaults; it ships its own Node, so it installs from PyPI and runs offline like the other tools.

## Options not chosen

- mypy: slower, and checks less by default.
- pyright from npm: needs Node installed beside uv, and is pinned outside `uv.lock`.
- black, isort and flake8: ruff does their work in one tool.
- PyPI's `gguf`: cannot read the model's file.

## Consequences

- `pyproject.toml` holds the ruff, basedpyright and pytest settings.
- [DEVELOPING.md](../DEVELOPING.md) gives the commands.
- `gguf` moves to a new llama.cpp commit only with the reference commit, under ADR-0002.
