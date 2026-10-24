# Guides

A guide shows how to do one task with C0ven. The list of guides is in the menu beside this page.

A guide does not explain how the server works or list its options. The [README](../../README.md) and the other
pages do that, and a guide links to them.

## Writing a guide

A guide is a Markdown file in this folder. Write it as you would for GitHub.

- The first heading is the guide's title. The site's menu shows it. You do not add the guide to any list.
- Start with what the reader needs. Then give the steps, in order. Stop when the task is done.
- Link to other pages with relative paths, such as `../MODEL_FILES.md#getting-the-files`.
- Put images in this folder, next to the guide.

To see the site on your computer:

```bash
uv run python -m zensical serve
```

Open <http://localhost:8000>. The site updates when you save a file. The build fails if a link points to a page or
heading that does not exist. The site is published when the change is merged to `main`.
