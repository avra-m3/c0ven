# ADR-0014: how work is done

**Status:** accepted, 2026-10-09; rewritten 2026-10-14. Item 5 amended 2026-10-18.
**Revisit when:** a rule costs more than it saves.

## Decision

1. Work is planned in GitHub issues.
2. ADRs record major architectural decisions and are immutable. A new ADR is checked against every earlier one; an
   earlier ADR it conflicts with is marked *superseded by* the new one, with a link.
3. The README is for people running the engine: readable, with charts, holding what they are likely to need.
4. A comment never holds what belongs in `docs/`.
5. Measurements follow [how the numbers were taken](../measurements/MEASUREMENTS.md#how-the-numbers-were-taken). A
   measurement adds its row to [MEASUREMENTS.md](../measurements/MEASUREMENTS.md) and replaces its estimate in the
   same change.
6. Code used only to measure is not committed. Its record describes it well enough to rewrite it.
