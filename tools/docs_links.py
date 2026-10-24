"""A Markdown extension for the published site (zensical.toml): links to files it does not hold go to GitHub.

The pages link the code and other files by relative paths, as GitHub reads them. The site holds only the pages, so
such a link is rewritten to the file's page in the repository on GitHub. Zensical keeps rendered pages in .cache/,
which a change to this file does not clear: delete it.
"""

from __future__ import annotations

import posixpath
from pathlib import Path
from xml.etree.ElementTree import Element

from markdown import Extension, Markdown
from markdown.treeprocessors import Treeprocessor
from zensical.extensions.context import ContextPreprocessor

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/avra-m3/c0ven"
# What the site holds: the paths zensical.toml's exclude_docs keeps.
SITE = ("README.md", "AGENTS.md", "docs")


class OutsideLinks(Treeprocessor):
    def run(self, root: Element) -> None:
        context = ContextPreprocessor.from_markdown(self.md)
        if context is None:
            return
        here = posixpath.dirname(context.page.path)
        for el in root.iter():
            attr = {"a": "href", "img": "src"}.get(el.tag)
            url = el.get(attr) if attr else None
            if attr and url and (new := self.rewrite(url, here)):
                el.set(attr, new)

    def rewrite(self, url: str, here: str) -> str | None:
        """The GitHub address for a relative link to a file the site does not hold; None to leave the link."""
        if url.startswith(("#", "/")) or ":" in url.split("/", 1)[0]:
            return None
        path, hash_, frag = url.partition("#")
        target = posixpath.normpath(posixpath.join(here, path))
        held = target.split("/", 1)[0] in SITE
        if held or target.startswith("../") or not (ROOT / target).exists():
            return None
        kind = "tree" if (ROOT / target).is_dir() else "blob"
        return f"{REPOSITORY}/{kind}/main/{target if target != '.' else ''}{hash_}{frag}"


class DocsLinks(Extension):
    def extendMarkdown(self, md: Markdown) -> None:  # noqa: N802 (Python-Markdown's name)
        # Priority 0: after the inline links are made.
        md.treeprocessors.register(OutsideLinks(md), "docs_links", 0)


def makeExtension() -> DocsLinks:  # noqa: N802 (Python-Markdown's name)
    return DocsLinks()
