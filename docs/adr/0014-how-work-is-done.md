# ADR-0014: how work is done

**Status:** accepted, 2026-10-09; rewritten by the owner, 2026-10-14. Item 5 amended 2026-10-18: the
measurement records are kept privately, and [RESULTS.md](../RESULTS.md) gives their questions and answers.
**Revisit when:** a rule costs more than it saves.

The project is in active development: PRs are not required for now.

## Decision

1. Work is planned in GitHub issues.
2. ADRs record major architectural decisions and are immutable. A new ADR is checked against every earlier one; an
   earlier ADR it conflicts with is marked *superseded by* the new one, with a link.
3. The README is for people running the engine: readable, with charts, holding what they are likely to need.
4. A comment never holds what belongs in `docs/`.
5. Measurements on the machine follow [RESULTS.md#how-the-numbers-were-taken](../RESULTS.md#how-the-numbers-were-taken) and are committed as records in
   `docs/measurements/`. A measurement replaces its estimate in the same commit.
6. Code used only to measure is not committed. Its record describes it well enough to rewrite it.
