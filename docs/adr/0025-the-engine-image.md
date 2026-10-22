# ADR-0025: one image, defined by a `Dockerfile` at the root, builds, tests and runs the engine

**Status:** accepted, 2026-10-14. Supersedes item 1 of [ADR-0024](0024-repository-layout.md) in part: the layout
gains a `Dockerfile` at the root. Amended 2026-10-16: decisions 6 and 7 add the `serve` target. Amended 2026-10-18:
the `serve` target's directory is `/opt/c0ven`.
Item 7 superseded in part by [ADR-0043](0043-publish-only-the-serving-image.md): no `engine` image is pushed
beside `serve`.
**Revisit when:** the engine's run in service needs something a build image should not carry.

## Context

- [ADR-0002](0002-pin-the-toolchain.md) pins one set of versions for building, probing and running the engine.
- Builds and tests on the cards need the same image, rebuilt the same way, wherever they run.
- The engine runs in the same image.
- [ADR-0024](0024-repository-layout.md)'s layout has no place for a container definition.
- The core finds its kernels and its routing profile by absolute paths compiled in at build time: `ENGINE_KERNEL_DIR`
  is `<build>/kernels` ([cmake/kernels.cmake](../../cmake/kernels.cmake)), and `ENGINE_ROUTING_PROFILE` is under the
  source tree ([core/CMakeLists.txt](../../core/CMakeLists.txt)). A build only runs where it was built.
- The packages are top-level and not installed ([ADR-0024](0024-repository-layout.md), `package = false` in
  `pyproject.toml`), so `python -m frontend` runs from the checkout. The front end starts the engine as its child with
  its own interpreter, and nothing under `engine/` or `frontend/` opens a file for writing.

## Decision

1. A `Dockerfile` at the repository's root defines the image the engine is built, tested and run in.
2. The image installs exactly ADR-0002's versions. Ubuntu packages come from a dated archive snapshot; release
   downloads are checked against ADR-0002's sha256 sums.
3. It runs as a non-root user. The render group, the memory lock and the mounts are given at run time.
4. An upgrade under ADR-0002 changes the `Dockerfile` in the same commit.
5. Scripts that check a container are measurement code: not committed, described in the measurement record.
6. The `Dockerfile` has two targets. `engine` is the image of decisions 1-4, where the engine is built and tested.
   `serve` starts `FROM engine`, so it holds exactly ADR-0002's versions and pins none of its own. It copies the code
   the engine runs from (the core, its kernels and headers, the CMake files, the Python packages and their lock) to
   `/opt/c0ven`, its working directory, builds the core there in CMake's default build type (Release) into
   `/opt/c0ven/build`, and makes the venv at `/opt/c0ven/.venv` with `uv sync --locked --no-dev`.
   Root owns all three; nothing moves them after the build. It sets `ENGINE_LIBRARY` to
   `/opt/c0ven/build/core/libengine.so`, puts the venv's `bin` first on `PATH` and sets
   `PYTHONDONTWRITEBYTECODE=1`. Its entrypoint is `python -m frontend`, with no command, so the arguments of
   `docker run` are the front end's flags. It runs as `ubuntu` (uid 1000) and writes nothing to its root filesystem,
   which may be mounted read-only.
7. CI builds `serve` after the engine's checks pass, in the same run, on every branch, and checks it: the entrypoint,
   the user, the source commit in `org.opencontainers.image.revision`, `--help` on a read-only root filesystem, and
   the core and its kernels loading from the image's paths. On `main`, after that check, it pushes `serve` to
   `ghcr.io/<owner>/<repository>/serve`, tagged `main` and `sha-<short commit>`, beside
   [ADR-0028](0028-ci-without-a-card.md)'s `engine` image.

## Options not chosen

- A `container/` folder with the Dockerfile and a check script: the check is measurement code
  ([ADR-0014](0014-how-work-is-done.md), decision 6), which leaves the Dockerfile alone in it.
- A separate image to run the engine, from a second Dockerfile: two images drift, and ADR-0002 asks for one set. A
  target built `FROM engine` in this file cannot drift from it: the versions are written once, and CI builds both
  from one file in one run.
- Moving the build after it ran: the core's compiled paths would point at nothing.

## Consequences

- ADR-0024's layout gains `Dockerfile` at the root.
- `docker build` of the whole file builds `serve`, which needs the repository as its context. The toolchain alone is
  `--target engine` ([README.md](../../README.md#building-from-source)).
- The core is built twice in a CI run: once in the container the checks run in, once in `serve`.
