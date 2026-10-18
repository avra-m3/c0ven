# ADR-0021: builds and probes run in a measurement container of our own on the machine

**Status:** accepted, 2026-10-11, by the owner; confirmed the same day by
[P51](../precursor/P51-measurement-container.md) ([MACHINE.md](../MACHINE.md#the-measurement-container)). Supersedes [ADR-0020](0020-the-sandbox.md). ADR-0002's pinned versions stand.
**Revisit when:** the container cannot see both B60s, cannot hold an unlimited memory lock, or the owner wants the
work back in the existing sandbox.

## Context

- **ADR-0020 put the work in the existing sandbox.** It has no APT access, so every
  system package waits for the owner to add it to the image ([P47](../precursor/P47-sandbox-dependencies.md)). Its memory
  lock is 8 MiB ([MACHINE.md](../MACHINE.md#the-sandbox)). Every sandbox plan waited on that.
- **ADR-0002's whole set has already run in a throwaway container on the machine** ([P23](../precursor/P23-adr-toolchain.md)):
  `ubuntu:26.04` by digest, and the Level Zero loader saw both B60s through it.
- **A cloud session cannot reach the machine.**

## Decision

1. **Builds, probes and the reference runs happen in one measurement container on the machine,**
   built from ADR-0002's set and set up by [P51](../precursor/P51-measurement-container.md).
   It reaches the cards through `/dev/dri` (not `--privileged`), has `--ulimit memlock=-1:-1`, mounts the model files
   read-only, and keeps its scratch in a named Docker volume. Everything it adds to the host carries the label
   `project=flash-next-b60`, and three commands remove it.
2. **Agents reach it only through the owner's own access to the machine.** A cloud session never does.
3. **Agents never change the host.** They may read it, run commands in containers and create throwaway
   containers, but nothing they do outlasts those containers. Freeing the cards (stopping the other models on
   them), pinning large amounts of host RAM, and anything that needs `sudo` are host changes: the owner does them,
   or says so each time.
4. **System packages go into the container's image,** which the agents build themselves, at ADR-0002's versions.
   Python packages are added with `uv`, pinned in the project that uses them.
5. **The container's versions and image digest are recorded** in every measurement record.

## Options not chosen

- **The existing sandbox (ADR-0020):** every package waits on the owner, and its memory lock stops P27 and the engine.

## Consequences

- P47 is dropped; its questions are answered by P51 inside the container.
- Every plan marked "the sandbox" now means this container.
- The container stays on the machine until the owner asks for it to go.
