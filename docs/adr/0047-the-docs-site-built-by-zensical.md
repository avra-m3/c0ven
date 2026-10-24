# ADR-0047: the docs are published as a site built by Zensical, with the guides in docs/guides/

**Status:** draft, 2026-10-24; not accepted.
**Revisit when:** Zensical changes how it reads `mkdocs.yml`-style projects, stops accepting the repository's root
as `docs_dir`, or gains the plugins that would replace `tools/docs_links.py`.

## Context

- The [Diátaxis](https://diataxis.fr) scheme sorts documentation by what the reader is doing. Tutorials and how-to
  guides serve action; explanation and reference serve knowledge.
- `docs/` holds explanation and reference ([DESIGN.md](../DESIGN.md), [CACHING.md](../CACHING.md), the specs, the
  measurements). The steps for an action sit in the README (the server's run, the sleep and wake setup) and in
  [MODEL_FILES.md](../MODEL_FILES.md), among facts a reader does not need to act.
- The pages are Markdown, read on GitHub. Their links are relative paths to other pages, to headings by GitHub's
  anchors, and to the code.
- [ADR-0014](0014-how-work-is-done.md) item 3 makes the README the page for people running the engine.
  [ADR-0024](0024-repository-layout.md) fixes the repository's layout.
- Material for MkDocs, the most used theme of MkDocs, takes critical fixes only, until 2027-05-05. MkDocs 1.6.1
  dates from 2024, and MkDocs 2.0 does not build 1.x projects. [Zensical](https://zensical.org) is Material's
  successor from the same team. It reads the same configuration and is pre-1.0.

## Decision

1. The README, [AGENTS.md](../../AGENTS.md) and `docs/` are published to GitHub Pages as one site, built by
   Zensical from `zensical.toml` at the root. The pages stay Markdown written for GitHub; nothing in a page is
   for the site alone.
2. The site's navigation is made from the folders and each page's first heading. Nothing lists the pages: a new
   page in `docs/` is on the site.
3. `docs/guides/` holds the how-to guides and tutorials: pages whose reader has a goal and wants the steps. The
   rest of `docs/` is explanation and reference. [docs/guides/index.md](../guides/index.md) says how to write a
   guide and preview the site.
4. `zensical.toml` turns on the Markdown extensions whose output matches GitHub's, and GitHub's heading anchors.
   `tools/docs_links.py`, a Python-Markdown extension, sends a link to a file the site does not hold (the code) to
   that file on GitHub.
5. The build is strict: a link to a missing page or heading fails it. The `site` workflow builds the site for each
   pull request that changes it and publishes it from `main`.
6. The site's colours are those of `docs/images/logo.svg`: #1F3B34, #143229, #2F786A, #44AE8E and the gold
   #F5B83D, in `docs/stylesheets/logo-colours.css`. A change to the logo's colours changes the stylesheet in the
   same pull request.
7. A guide gives the steps and links the page that holds a fact the reader may need. A fact has one home
   ([AGENTS.md](../../AGENTS.md)): when a guide takes steps from the README or `docs/`, they leave that page and a
   link stays. The README keeps a short section for each guide that took its steps, so existing links to it hold.
8. Zensical is pinned in the `site` dependency group, which the `dev` group includes and the image leaves out.

## Options not chosen

- Material for MkDocs on MkDocs 1.6.1: established, but in maintenance. Zensical reads the same configuration, so
  the choice can be undone.
- Sphinx with MyST: maintained and strict about links, but every page must sit in a `toctree`, written in
  directives that GitHub shows as code.
- Docusaurus: a Node.js toolchain beside uv's ([ADR-0002](0002-pin-the-toolchain.md)).
- A generator written here: the same work as a framework's, kept by this project.
- Plain HTML guides in a `guides/` folder at the root: each page repeats its frame, the index is kept by hand,
  GitHub shows the HTML's source, and links into `docs/` break on the site.
- `docs_dir = "docs"`: the pages' links to the README and AGENTS.md fail the build, and the README is not the
  home page.

## Consequences

- The first guide is [sharing the cards through llama-swap](../guides/share-the-cards.md), moved out of the
  README.
- The repository's Pages source must be set to GitHub Actions.
- `docs_dir` is the repository's root, with `exclude_docs` keeping only the site's files. Zensical's
  documentation does not list a root `docs_dir` as supported.
- A link to a heading that GitHub numbers as a repeat (`#tests-1`) needs an explicit anchor in the page, as
  Python-Markdown numbers repeats differently.
