# ADR-0043: CI publishes only the serving image

**Status:** accepted, 2026-10-20. Supersedes item 6 of
[ADR-0028](0028-ci-without-a-card.md), and item 7 of [ADR-0025](0025-the-engine-image.md) where it puts the `serve`
image beside an `engine` image on GHCR.
**Revisit when:** a developer needs the development image without building it.

## Context

- ADR-0025 and ADR-0028 push two images: `ghcr.io/avra-m3/c0ven/engine`, the image the engine is built and tested
  in, and `ghcr.io/avra-m3/c0ven/serve`, the one that runs the server ([ADR-0025](0025-the-engine-image.md)). Both
  appear in the repository's packages, and only `serve` runs the server: `engine` has no entrypoint and no engine
  inside.
- The development image builds from the `Dockerfile` with `docker build --target engine`
  ([DEVELOPING.md](../DEVELOPING.md#build)).

## Decision

1. CI builds the `engine` image on the runner and runs its checks in it, as ADR-0028 items 2-5 say, and does not push
   it.
2. CI pushes only `serve` to GHCR, from `main` and from version tags, after the checks pass.

## Options not chosen

- Keeping the `engine` image under a name that says it is for development: the package list would still show two
  images, and developers build it from the checkout they work in.

## Consequences

- The development image is built locally. DEVELOPING.md says how.
- [ADR-0042](0042-version-numbers.md)'s releases publish the serving image under the version.
