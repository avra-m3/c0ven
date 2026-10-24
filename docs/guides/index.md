# Guides

Each guide is for doing one thing with C0ven. The guides are listed beside this page. What the server is, how it
works, and what its flags and API are, is in the [README](../../README.md) and the other pages of the site:
explanation and reference. The guides link there and do not repeat it.

## Writing a guide

A guide is one Markdown file in this folder, written as for GitHub. Its first heading names the goal and is its
title in the site's navigation, which lists every page in `docs/` by itself. The guide lists what the reader needs,
gives the steps in order, and ends when the goal is met. A step that needs a fact the reader may lack links the
page that holds it, by its relative path and GitHub's anchor for the heading. A picture goes beside the guide.

To see the site as it will be published, with the guide in it:

```bash
uv run python -m zensical serve
```

It serves the site at <http://localhost:8000> and rebuilds it as files change. The build stops on a link to a page
or heading that does not exist. Merged to `main`, the guide is published.
