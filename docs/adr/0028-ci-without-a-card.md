# ADR-0028: CI on GitHub's runners builds the image and the engine, runs the checks a runner can, and publishes the image

**Status:** accepted, 2026-10-16. Supersedes item 1 of
[ADR-0024](0024-repository-layout.md) in part: the layout gains `.github/workflows/`. Item 6 superseded by
[ADR-0043](0043-publish-only-the-serving-image.md): the image is not pushed.
**Revisit when:** a runner with a B60 is added, or a level-1 test meets a runner without AVX-512.

## Context

- GitHub's hosted runners have no Intel GPU, and Level Zero has no CPU device
  ([ADR-0001](0001-level-zero-from-scratch.md)). The model is 83.6 GB ([MODEL_FILES.md](../MODEL_FILES.md)). So no
  kernel test and no test that reads the model can run there, and every one of the core's tests is one or the other.
- The Python half's host tests need neither ([DEVELOPING.md](../DEVELOPING.md#tests-1)). [ADR-0026](0026-python-tooling.md)
  asks every change to pass `ruff format`, `ruff check`, `basedpyright` and `pytest`. In the engine image on a host
  with no card or model, pytest runs the host tests and skips those that read the model.
- `ocloc` compiles for `bmg-g21` without a card. In the engine image on a host with no card, the build compiles every
  kernel module and links every test, warning-free with `-DENGINE_WERROR=ON`. The image is about 1.9 GB.
- Without a card, a C++ GPU test skips, and fails instead when `ENGINE_EXPECT_GPUS` is set
  ([DEVELOPING.md](../DEVELOPING.md#tests)). A missing card is never mistaken for a broken kernel.
- ggml is built with AVX-512 ([cmake/ggml.cmake](../../cmake/ggml.cmake)). GitHub does not promise a CPU model for
  its hosted runners, so a test that runs ggml's CPU code there may stop on an illegal instruction.
- The [Level Zero](../measurements/LEVEL_ZERO.md) packages ADR-0002 pins carry no null driver (`libze_null.so`); `dpkg -L libze1`
  lists the loader and its validation and tracing layers only.

## Decision

1. `.github/workflows/ci.yml` runs on every push and on demand, on a GitHub-hosted Ubuntu runner.
2. It builds the image from the root `Dockerfile`, then builds the engine in a container of it, with the checkout
   mounted read-only and `-DENGINE_WERROR=ON`. Every kernel is compiled.
3. It runs ADR-0026's four Python checks, with the venv and the caches outside the checkout. pytest finds the core
   just built through `ENGINE_LIBRARY`.
4. It runs the core's tests without the `gpu` label, and fails if one fails.
5. It runs the `gpu` tests and fails unless every one skips. It runs `ze_smoke_test` with `ENGINE_EXPECT_GPUS=2` and
   fails unless the test fails. Together these check the harness: without a card, a GPU test neither passes nor
   crashes, and the machine's runs cannot mistake a missing card for a skip.
6. On `main`, after those steps, it pushes the same image to `ghcr.io/<owner>/<repository>/engine`, tagged `main`
   and `sha-<short commit>`. An image that cannot build the engine is never pushed.
7. Actions are pinned by commit, with their release tag in a comment.
8. Levels 2-7 run by hand on a machine with two B60s, as [DEVELOPING.md](../DEVELOPING.md#tests) says.

## Options not chosen

- Level Zero's null driver: the pinned packages do not ship it, and it answers calls without running kernels, so a
  kernel test on it would fail or check nothing.
- An OpenCL CPU device (PoCL, Intel's CPU runtime): the engine loads `bmg-g21` binaries through Level Zero and its
  kernels use Intel's sub-group and DPAS builtins ([ADR-0003](0003-kernel-language.md)). A CPU device would run other
  code.
- Building the image and testing in separate jobs: the image would cross between them through a registry or
  an artifact, and a failing branch would publish an image.
- Pushing an image from every branch: the registry would fill with images nobody pulls.

## Consequences

- ADR-0024's layout gains `.github/workflows/`. ADR-0025 stands: CI builds and publishes the image its `Dockerfile`
  defines.
- A level-1 test added without the `gpu` label, or a pytest test without the `gpu` and `model` markers that needs
  neither, runs in CI with no change to the workflow.
