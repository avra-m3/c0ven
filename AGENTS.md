# Working rules

For people and agents alike.

- **Changes:** as pull requests, one small, coherent change each, titled by what it does. The owner reviews them.
- **Decisions:** [ADRs](docs/adr/README.md) bind and are immutable. [ADR-0014](docs/adr/0014-how-work-is-done.md)
  says how work is done.
- **Layout:** [ADR-0024](docs/adr/0024-repository-layout.md) says where code goes. A change that departs from it
  first needs a new ADR that supersedes it.
- **No Strata code** ([ADR-0001](docs/adr/0001-level-zero-from-scratch.md)). Specs describe its behaviour in prose,
  tables and diagrams, citing its source at a pinned commit.

## Writing docs

- A page in `docs/` is a reference for users or developers, and the one source of truth for its subject. The
  README's [Documentation](README.md#documentation) list names each page's subject. A new page needs a subject no
  page holds.
- Plans, hypotheses, status and designs for work not yet built go in
  [GitHub issues](https://github.com/avra-m3/c0ven/issues) ([ADR-0014](docs/adr/0014-how-work-is-done.md)
  item 1), not in `docs/`. Once the work is built, the pages it changes describe it, and each decision it made is
  an ADR.
- Write reference, not narrative. Say what is, in the present tense. Leave out how a page or a fact came to be:
  who wrote it, what was tried first, what a review found. A date belongs next to the value it qualifies, not in the
  story of the page.
- Plain words, short sentences. Simple English over technical shorthand where both say the same.
- No codenames. Name the thing, or link the record, ADR or issue that holds it: not "P27 measured", "H9" or
  "stage 5". Names a reader types or searches for stay: files, functions, kernels, tests, flags, prompts, ADR
  numbers, commits.
- Each measured number links the row of [docs/measurements/MEASUREMENTS.md](docs/measurements/MEASUREMENTS.md) it comes from.
- An unmeasured number is labelled *estimate*, with its method.
- One home per fact; elsewhere, link to it: Level Zero calls in [LEVEL_ZERO.md](docs/measurements/LEVEL_ZERO.md), the model's and
  Strata's behaviour in [docs/spec/](docs/spec/).

## Writing code comments

- A comment says what the code cannot: why it is this way, an invariant, a unit, a layout or a contract. One or two
  lines is usual, four at most. A file's header says what the file is for in at most five lines. A fact that does
  not fit goes next to the line it concerns, or in a doc the comment links.
- No measurements, timings, dates or the story of an experiment in a comment. If the reason rests on a measurement,
  give the reason in a short clause and the record's path.
- Do not restate the code, describe earlier versions, or use codenames.
- Public interfaces (`include/engine_api.h`, public Python functions and classes) document their contract in full:
  each field or argument, units, ownership and errors. Tersely.

## Avoid repetitive structure

In docs and comments alike, repeated shapes read as machine-made
([Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing)). Check drafts for:

- Lists for non-parallel items; prose suits connected points. Bold lead-ins only in reference lists, like this page's.
- Threes by habit. Give as many items as exist.
- Denying unmade claims: "not just X, but Y", "Y rather than X".
- "Serves as", "stands as", "features". Write "is".
- Trailing "-ing" commentary ("…, highlighting…"). A consequence that matters gets its own sentence and evidence.
- Synonyms for one thing.
- Recaps, "key takeaways", closing outlooks.
- Bold beyond definitions and warnings.
- Frequent em dashes.
- Title-case or skipped headings, horizontal rules, emoji.
- Sentences sharing one opening or rhythm.
