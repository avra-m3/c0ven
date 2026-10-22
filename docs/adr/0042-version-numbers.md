# ADR-0042: one version number for the project, from 0.0.1, kept in VERSION and raised by each pull request

**Status:** draft, 2026-10-20; not accepted.
**Revisit when:** the server's API and options are to stay stable between releases (version 1.0.0), or a part of the
project needs to ship apart from the rest.

## Context

- Without a version a user can name, the images CI publishes ([ADR-0028](0028-ci-without-a-card.md)) are tagged
  `main` and `sha-<short commit>` only, and a deployment names a commit.
- The front end, the engine process and the core are built and shipped together, in one image
  ([ADR-0025](0025-the-engine-image.md)). The core's C ABI has its own number, `ENGINE_API_VERSION` in
  [`include/engine_api.h`](../../include/engine_api.h), which the control plane checks when it loads the core.

## Decision

1. The project has one version, `MAJOR.MINOR.PATCH`. The first is 0.0.1.
2. A file `VERSION` at the root, holding only the version, is its one home. `pyproject.toml` declares the version
   dynamic, CMake's `project()` reads `VERSION`, and `python -m frontend --version` prints it.
3. Each pull request raises `VERSION` above its base's. Its author picks the number. While the major number is 0, a
   change to the server's options, its API or the model files it accepts raises the minor number; any other change
   raises the patch number.
4. The `version` workflow fails a pull request whose `VERSION` is not above its base branch's tip.
5. Each merge to `main` is a release. CI publishes its serving image tagged with the version, beside the `main` and
   `sha-` tags, and labels it with the version. CI fails instead of publishing a version already published from
   another commit.
6. `ENGINE_API_VERSION` stays separate. It counts changes to the C ABI, and a release need not change it.

## Options not chosen

- A version derived from git tags at build time (setuptools-scm and the like): the project is not installed as a
  package (`package = false`), and the serving image is built without the `.git` directory.
- Separate versions for the front end and the core: they are never shipped apart, and the ABI number already guards
  the boundary between them.
- Calendar versions: the dates say nothing about whether a release breaks a deployment's options.
- `version` in `pyproject.toml` as the home: a file holding only the version is one line to read and to change, for
  CMake, the Dockerfile and the workflows alike.
- Releases as git tags pushed by hand: a merged change would wait for someone to tag it before an image named its
  version.
- A bump chosen by CI from labels or commit messages: the author knows best what the change breaks.

## Consequences

- CMake fails to configure if `VERSION` does not hold `MAJOR.MINOR.PATCH`.
- Two open pull requests may raise `VERSION` to the same number. Once one merges, the other's check fails when it
  runs again; if it merges on an old check, CI on `main` fails rather than publish over the first.
- The images can be pulled by version: `ghcr.io/avra-m3/c0ven/serve:0.0.1`.
