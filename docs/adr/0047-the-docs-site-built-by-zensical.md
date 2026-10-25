# ADR-0047: the docs are published as a site built by Zensical, with the guides in docs/guides/

**Status:** draft, 2026-10-24; not accepted.
**Revisit when:** Zensical stops reading this configuration, stops allowing the repository's root as `docs_dir`, or
adds plugins that can do what `tools/docs_links.py` does.

## Context

- [Diátaxis](https://diataxis.fr) sorts documentation by what the reader wants. How-to guides and tutorials are
  for doing a task. Explanation and reference are for understanding.
- `docs/` holds explanation and reference. Steps for tasks are in the README and in
  [MODEL_FILES.md](../MODEL_FILES.md), mixed with facts the reader does not need for the task.
- The pages are Markdown, and people read them on GitHub. They link to other pages and to code by relative paths,
  and to headings by GitHub's anchors.
- [ADR-0014](0014-how-work-is-done.md) item 3 makes the README the page for people who run the engine.
  [ADR-0024](0024-repository-layout.md) sets the repository's layout.
- Material for MkDocs gets only critical fixes, until 2027-05-05. The last MkDocs 1.x release, 1.6.1, is from
  2024. MkDocs 2.0 cannot build 1.x projects. [Zensical](https://zensical.org) is made by the Material team to
  replace it. It reads the same configuration. Its version is below 1.0.

## Decision

1. Zensical builds the README, [AGENTS.md](../../AGENTS.md) and `docs/` into one site on GitHub Pages. Its
   configuration is `zensical.toml`, at the root. The pages are plain Markdown for GitHub, with nothing added for
   the site.
2. The site's menu comes from the folders and from each page's first heading. No file lists the pages. A new page
   in `docs/` is on the site.
3. `docs/guides/` holds the how-to guides and tutorials. The rest of `docs/` is explanation and reference.
   [docs/guides/index.md](../guides/index.md) says how to write a guide and how to see the site locally.
   `docs/showcase/` holds things coding agents have made, each page with the steps to make it.
4. `zensical.toml` turns on only the Markdown extensions that give the same result as GitHub, and makes heading
   anchors the way GitHub does.
5. The site does not hold the code. `tools/docs_links.py`, a Python-Markdown extension, changes each link to a
   file the site does not hold into a link to that file on GitHub.
6. The build is strict: it fails on a link to a page or heading that does not exist. The `site` workflow builds
   the site for each pull request that changes it, and publishes the site from `main`.
7. The site uses the colours of `docs/images/logo.svg`: #1F3B34, #143229, #2F786A, #44AE8E and #F5B83D. They are
   in `docs/stylesheets/logo-colours.css`. A pull request that changes the logo's colours changes this file too.
8. A guide gives steps and links to the page that holds each fact. Each fact is in one place
   ([AGENTS.md](../../AGENTS.md)). When steps go into a guide, they are removed from the README or `docs/`, and a
   link to the guide stays there under the same heading.
9. Zensical is pinned in the `site` dependency group. The `dev` group includes it. The image does not.

## Options not chosen

- Material for MkDocs on MkDocs 1.6.1: widely used, but it gets only critical fixes. Zensical reads the same
  configuration, so a move back is small.
- Sphinx with MyST: maintained and strict about links. But each page must be listed in a `toctree` directive,
  which GitHub shows as code.
- Docusaurus: it needs Node.js, a second toolchain next to uv ([ADR-0002](0002-pin-the-toolchain.md)).
- A site builder written for this project: the project would maintain what a framework already does.
- Plain HTML guides in a root `guides/` folder: each page repeats the same header, the index is kept by hand,
  GitHub shows the HTML source, and links into `docs/` do not work on the site.
- `docs_dir = "docs"`: links from `docs/` to the README and AGENTS.md fail the build, and the README cannot be the
  home page.

## Consequences

- [docs/guides/share-the-cards.md](../guides/share-the-cards.md) holds the steps for sharing the cards through
  llama-swap.
- The repository's Pages source must be GitHub Actions.
- `docs_dir` is the repository's root, and `exclude_docs` leaves out everything but the site's pages. Zensical's
  documentation says the root is not allowed, though the build works.
- GitHub gives a repeated heading the anchor `-1`, and Python-Markdown gives it `_1`. A link to a repeated heading
  needs an explicit anchor in the page, such as `<a id="tests-1"></a>`.
