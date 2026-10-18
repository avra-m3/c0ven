# ADR-0020: the engine is built and probed in the existing sandbox

**Status:** superseded by [ADR-0021](0021-the-measurement-container.md) on 2026-10-11. Accepted 2026-10-10, by the owner. Supersedes item 3 of [ADR-0002](0002-pin-the-toolchain.md) (one
container image, built by P24). ADR-0002's pinned versions stand.
**Revisit when:** the sandbox cannot carry ADR-0002's set, or cannot lift the memory-lock limit the engine needs.

## Context

- **A sandbox already exists on the machine:** a shared development container with Docker inside it and both B60s
  ([MACHINE.md](../MACHINE.md#the-sandbox)). Agents work in it.
- **It has no APT access.** System packages are added to the container's image by the owner, so they must be known
  in advance. Python packages are added with `uv add`, which works.
- **ADR-0002's set is already chosen and tested** ([P23](../precursor/P23-adr-toolchain.md), in a throwaway container).
  Building a second image for it (P24) duplicates the sandbox.

## Decision

1. **Builds, probes and the reference runs happen in the sandbox.** No new toolchain image is built.
2. **The system packages the plans need are listed in advance** ([P47](../precursor/P47-sandbox-dependencies.md)), with
   versions from ADR-0002, and the owner adds them to the sandbox's image. A plan that needs a package not on the list
   adds it to the list first, and waits for it.
3. **Python packages are added with `uv`,** pinned in the project that uses them.
4. **The versions in the sandbox are recorded** in every measurement record, as before.

## Consequences

- P24 is dropped. P40-P44 shrink to one scaffold plan ([P49](../precursor/P49-scaffold.md)).
- The sandbox's memory-lock limit decides whether P27 can pin ~50 GB. The engine needs it unlimited.
- CI (P43) is deferred: there is no image to pull by digest. The engine's GPU tests run in the sandbox.
