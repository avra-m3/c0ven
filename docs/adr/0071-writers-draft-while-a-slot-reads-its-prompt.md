# ADR-0071: writing slots draft while another slot reads its prompt

**Status:** draft, 2026-11-09; not accepted. Supersedes in part the clause of
[ADR-0053](0053-writers-rows-ride-in-a-prompt-chunk.md)'s item 1 that ties its rows to "several writers in a decode
step".
**Revisit when:** an ABBA shows the writers' tokens beside a reading slot no higher, or the reader's prompt rows a
second or its first token worse, than with plain steps; or a writer's row in a prompt chunk can carry its drafts.

## Context

- A decode step of several writing slots ran as a draft round only while no slot had prompt left to read, and after a
  plain step of several the drafter made no drafts while a slot read. The rule came with rounds of several writers,
  measured with the MTP layer and no slot reading
  ([the record](../measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers)). No record measures writers
  beside a reading slot.
- The scheduler gives the writers time, not steps: after each prompt step, decode is owed `decode_share` (0.5) of its
  time, and the next prompt step waits until the decode plans' times have paid it
  ([ADR-0017](0017-scheduling-several-requests.md), item 1). Each plan's time is the sum of its timing records. A
  round's record covers its window, as a step's covers its step.
- A round takes longer than a plain step and gives more tokens. A 6-row step takes 34.54 ms and a 2-row step 19.25 ms
  ([widths](../measurements/MEASUREMENTS.md#2026-10-28-step-widths-3-and-6),
  [rows](../measurements/MEASUREMENTS.md#2026-10-19-four-rows-a-step)). Two writers with 2 drafts each fill 6 rows.
  A DFlash 2 pass costs 4.0 ms a window ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-on-the-cards)).
- A deployment drafting with DFlash 2 made no round while any slot read, and a slot read for a large part of the day
  (counters in the pull request that brings this ADR).

## Decision

1. A decode step whose rows are all writing slots' runs as a draft round when some slot has drafts, whether or not
   another slot reads its prompt. After a plain step, the drafter makes drafts for each writing slot whether or not
   another slot reads. The same holds after a round.
2. A step of several slots that gives one of them its second token stays a plain step, as before: that token already
   waits for the prompt steps after the first, and a round would lengthen the step that ends the wait.
3. A step whose rows include a slot's prompt token stays a plain step.
4. A writer's row in a prompt chunk ([ADR-0053](0053-writers-rows-ride-in-a-prompt-chunk.md)) gets no drafts, as
   before.
5. The rule is the same for DFlash 2 and the MTP layer.
6. The core counts the plain steps whose rows all wrote (`undrafted_steps`), and the rounds, run while another slot
   read its prompt: `engine_stats.reading_steps` and `reading_rounds` (ABI version 21). `/metrics` reports
   `reading_round_share`, the rounds' part of the two.

7. The core writes a timing record for what follows a plan's step or round: the sampling and the draft pass
   (`ENGINE_TIMING_DRAFT`, rows 0). It is part of the plan's time, so the owed time counts it.
8. While a slot has prompt left to read, the time a decode plan runs past what is owed is kept as credit: the next
   prompt step's debt is paid down by it. With no slot reading, the owed time stays at 0 or more, as before.

## Why

The writers' share of time beside a reading slot is fixed by the scheduler, so the question is only how many tokens
that time gives. Rounds of several writers gave more tokens a second than plain steps with the MTP layer
([the record](../measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers)), and a DFlash 2 window gives
more tokens than an MTP one ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-on-the-cards)). Rounds of
several writers with DFlash 2 beside a reader are judged by the ABBA the pull request lists. The second-token rule costs
one plain step per request, so it stays until something measures it. One rule for both drafters keeps one path through
the step; the reason for the old rule did not depend on the drafter.

The first ABBA on the cards (two cards, 4 slots, DFlash 2, uncached prompts) showed the mechanism working and a cost:
`reading_round_share` was 0.88-0.92, two writers beside a 32k prompt went from 43.8 to 58.6 tokens/s, and the reader's
prompt rate fell about 8% (1,036 to 957 tokens/s). The writers took more than their share of time. Two places in the
code let them. A plan was charged only its step or round, not the draft pass after it (beside a reader there was none before
this ADR). And time past the debt was not counted: the plan that paid the debt and the plans queued behind it (up to
2) ran before the next prompt step, and the scheduler cut the owed time at 0. Items 7 and 8 close both, so the writers get `decode_share` of the prompt steps' time and the gain comes from
rounds giving more tokens in it.

## Options not chosen

- DFlash 2 only, the MTP layer keeping the old rule. The MTP record that set the rule did not measure writers beside a
  reader either, and two rules would need two sets of checks.
- Dropping the second-token rule too. Nothing measures it, and it is one step per request.

## Consequences

- The writers' plans beside a reader are rounds of up to 8 rows instead of plain steps of 2 or 3. The plan that pays
  the debt, and the plans already queued behind it (the native adapter keeps up to 2), run before the next prompt step:
  up to about three writers' plans, about 104 ms of 6-row rounds against 58 ms of 2-row steps (*estimate*: three
  plans at the step times above). That time is credit against the next prompt step's debt, so over a run of prompt
  steps the writers get `decode_share` of the prompt steps' time, and no more.
- Every decode plan's time, with or without a reader, now includes the draft pass after it, about 4.0 ms. The cost
  model's decode step time (`decode_ms`) includes it too, and so does a prompt plan's decode part that it subtracts.
- A writer whose row rides in a prompt chunk loses its drafts, so the decode step in the same plan is a plain step.
  Drafting for those writers after the chunk is a follow-up.
