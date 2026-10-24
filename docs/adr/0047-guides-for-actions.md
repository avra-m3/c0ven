# ADR-0047: how-to guides live in guides/, apart from docs/

**Status:** draft, 2026-10-24; not accepted.
**Revisit when:** a guide needs more than a page of HTML can give, or the folder outgrows a flat list.

## Context

- The [Diátaxis](https://diataxis.fr) scheme sorts documentation by what the reader is doing. Tutorials and how-to
  guides serve action; explanation and reference serve knowledge.
- `docs/` holds explanation and reference ([DESIGN.md](../DESIGN.md), [CACHING.md](../CACHING.md), the specs, the
  measurements). The steps for an action sit in the README (the server's run, the sleep and wake setup) and in
  [MODEL_FILES.md](../MODEL_FILES.md), among facts a reader does not need to act.
- [ADR-0014](0014-how-work-is-done.md) item 3 makes the README the page for people running the engine.
  [ADR-0024](0024-repository-layout.md) fixes the repository's layout.

## Decision

1. `guides/` at the root holds the how-to guides and tutorials: pages whose reader has a goal and wants the steps.
   `docs/` holds explanation and reference only.
2. A guide is one HTML page with `guides/style.css`. There is no generator. `guides/index.html` lists the guides.
3. The stylesheet's colours are those of `docs/images/logo.svg`: #1F3B34, #143229, #2F786A, #44AE8E and the gold
   #F5B83D. A change to the logo's colours changes the stylesheet in the same pull request.
4. A guide gives the steps and links `docs/` for what the reader may need to know. A fact has one home
   ([AGENTS.md](../../AGENTS.md)): when a guide takes steps from the README or `docs/`, they leave that page and a
   link stays.
5. The README keeps a short section for each guide that took its steps, so existing links to it hold.

## Options not chosen

- A static site generator (MkDocs, Docusaurus): a toolchain to pin ([ADR-0002](0002-pin-the-toolchain.md),
  [ADR-0026](0026-python-tooling.md)) and a build to run in CI, for a handful of pages.
- Guides in Markdown: GitHub renders it with its own colours, so the logo's cannot apply.
- Guides in `docs/guides/`: `docs/` would hold two kinds of page, and the split is the point.

## Consequences

- The first guide is [sharing the cards through llama-swap](../../guides/share-the-cards.html), moved out of the
  README.
- Nothing publishes the folder yet. It is read from a checkout or served from any static host.
