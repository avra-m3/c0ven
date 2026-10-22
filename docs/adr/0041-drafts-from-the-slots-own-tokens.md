# ADR-0041: a lone writing slot drafts from its own tokens when that is worth more than the draft layer

**Status:** draft, 2026-10-19; not accepted. Would add a second source of drafts to
[ADR-0029](0029-the-engine-process-drafts-and-swaps.md)'s draft rounds.
**Revisit when:** the draft layer's acceptance on repeated text comes close to the lookup's, a window of 5-8 rows
costs about what one of 4 costs, or rounds for several slots take the lookup's drafts too.

## Context

- The draft layer proposes up to 3 drafts a round, each at probability 0.5 or more under the request's sampler, and
  a fourth was not worth its row ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-19-draft-experts-q8-and-four-drafts)).
- Answers that quote their prompt or their own earlier text (editing a file, answering from a document, summaries
  with repeated phrases) repeat long runs of tokens already in the slot. A copy of such a run is accepted far more
  often than the draft layer's drafts, and is not limited to 3.
- A window of one slot costs more with its rows, and the 5-8 row list much more than the 4-row one: in one run,
  16 ms for 1 row, 26 ms for 4 and 64 ms for 8 ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-19-lookup-drafts)). A
  long copy is worth its window only when nearly all of it is accepted.

## Decision

1. Each slot keeps where every run of three of its tokens ends (`core/api/lookup.cpp`). Before a round of one slot,
   the core matches the slot's last tokens, with the token sampled last, against earlier runs: of the last 4 places
   the last three tokens ended, the longest match of at least 12 tokens, up to 64, the newest of equal ones. The
   tokens that followed it, up to an image or video token, are the lookup's drafts.
2. The round takes the lookup's drafts instead of the draft layer's when they give more tokens a millisecond: the row
   sampled last plus the drafts expected to be accepted, over the window's time. The lookup's drafts are expected to
   be accepted one after another at the slot's own rate for them so far, the draft layer's at the slot's mean so far,
   both decayed by 0.9 a round. The lookup is offered as many drafts as the draft layer makes (3), and as many as a
   window holds (7).
3. A window's time is the running mean of the core's own decode steps and rounds of one slot with that many rows; a
   width not run yet is drawn on a line through the nearest widths run. Nothing is fixed for the machine.
4. Rounds of several slots draft from the draft layer only. The draft layer reads every committed row whichever
   source drafted, so its drafts stay ready.

## Why

Over eleven prompts with drafting, in five runs of this code against six of main's in alternating order, decode was 3.2%
faster (each run +2.3% to +3.5%, main's -0.9% to +0.2%): an edit of a Python file 21.0%, of a C++ file 6.1%, a question
on a 32,768-token document 9.5%, one on a 4,096-token document 4.0%, and the other prompts within -2.4% and +0.7%
([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-20-lookup-drafts-against-main)). Its lookup rounds always took 3 drafts: an 8-row
window was never worth its time. Taking the lookup's drafts whenever they were longer than the draft layer's, without
the comparison, made a summary 15% slower: long copies that failed early in the 8-row window.

## Options not chosen

- A fixed table of window times: the times change with the machine, the model's quant and the context length.
- Lookup drafts for rounds of several slots: each slot there gets at most 2 rows, which the draft layer fills well.
- Drafts from other slots' tokens or from a shared corpus: not measured.

## Consequences

- A slot's index costs a few words for each of its tokens. It grows as the prompt is read, a piece at a time, so
  the round after a prompt does not wait for the whole prompt to be indexed.
- Which source drafts depends on measured times, so the plans of a request differ from run to run. Tokens do not
  depend on how the rows were split into windows (docs/TESTING.md, level 6), so with `--deterministic` they repeat as
  before; with swaps they already differ from run to run.
