# ADR-0036: a private repository and its public copy

**Status:** accepted, 2026-10-17, by the owner, with the pull request that adds it. Supersedes item 1 of
[ADR-0024](0024-repository-layout.md) in part: the layout gains `private/` at the root.
**Revisit when:** people outside the project send changes to the public copy, or keeping the copy clean costs more
than a few lines a week.

## Context

- The engine, its docs and its measurement records are meant for anyone with two B60 cards. The same repository
  also holds how one machine is run: its name, the other services and models on it, its sandbox, how sessions reach
  it and share its cards, and the owner's approvals. The owner wants the first public and the second private.
- MACHINE.md describes that machine part by part, and the measurement records describe it run by run: its load, its
  versions and its timings. Each fingerprints it. The code and docs cite them for their numbers.
- Work lands on `main` many times a day, from sessions on the machine and in the cloud, with several branches in
  flight. A change that moves many files would conflict with all of them.

## Decision

1. This repository, `flash-next-b60`, stays private, and all work happens in it.
   [c0ven](https://github.com/avra-m3/c0ven) is its public copy.
2. `private/export.py` builds the copy from one commit. It leaves out `private/`, the paths in
   `private/exclude.txt` and every private block in a Markdown file. It opens every public block and applies
   `private/rewrites.tsv`. `private/README.md`, in the private repository, gives the syntax.
3. The test machine's page (MACHINE.md), the measurement records, their index (`tools/measurements.py`) and how they
   are taken (MEASURING.md) stay here. The copy has none of them.
4. Nothing moves. What is private stays where it is, marked.
5. The copy must match nothing in `private/denylist.txt`. A match that `private/exceptions.tsv` lists, with its
   reason, goes through. c0ven stays private until that list is empty.
6. c0ven's history is its snapshots. Each is one commit, by `avra-m3` with GitHub's no-reply address, whose message
   names the source commit. Nothing is committed to c0ven by hand.
7. Outside the records, details of the machine's setup go in a private block or under `private/`, and nowhere
   else.

## Options not chosen

- The engine in its own repository, kept here as a submodule: a change and the record that measures it become two
  commits in two repositories, and every branch in flight would need rebasing onto the new layout.
- `git subtree split` of a public folder: it copies each commit's message and author, and both name the machine and
  the sessions on it.
- The records published without their Machine state: the rest of each record still fingerprints the machine.

## Consequences

- ADR-0024's layout gains `private/`. Code outside it never reads it, and the build and the tests do not need it.
- AGENTS.md tells sessions where private details go. A new name the denylist does not know can still leak, so the
  denylist grows with the machine.
- The copy's code and docs cite records it does not have, so a public reader cannot check those numbers. A public
  account of the results, without the machine's details, is still to be written.
- A change proposed on c0ven is applied here, and reaches c0ven with the next snapshot.
