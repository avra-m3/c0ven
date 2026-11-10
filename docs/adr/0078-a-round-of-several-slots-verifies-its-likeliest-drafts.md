# ADR-0078: a round of several slots verifies its likeliest drafts

**Status:** draft, 2026-11-10; not accepted.
**Revisit when:** three requests writing at once, this change against the commit before it in alternating order,
give no more tokens a second in total; or the fitted estimate is compared with the acceptance measured and is off
by more than the sampler's probability alone.

## Context

- A round of several slots gave every slot the same number of drafts, at most its equal share of the step's rows.
  The number came from each slot's mean acceptance over its last rounds and the window times
  ([the record](../measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers)).
- Acceptance differs between slots and between drafts. DFlash 2's first draft is accepted 0.889 of the time and
  its seventh 0.525, scored offline
  ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-acceptance)). The sampler gives each draft a
  probability when it picks it, which the core used only as a bar (`draft_min_p`).
- A step holds 16 rows ([ADR-0077](0077-a-decode-step-holds-16-rows.md)), and DFlash 2's block pass holds 16 rows
  of all slots together. Three slots cannot each be drafted 7 tokens in one pass.
- vLLM ranks the drafts of all requests in a step by their chance of being accepted and verifies as many as give
  the most accepted tokens for the step's time. Its estimate of that chance is a logistic model of the draft's
  logits, fitted while it serves (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` and
  `acceptance_estimator.py` at commit `b2b5ae8`).

## Decision

1. The core keeps the probability the sampler gave each draft of the draft layer.
2. The chance that a draft is accepted, given that the slot's drafts before it were, is a logistic model of the
   log-odds of that probability: one slope, and an offset for each of the 7 places in a slot's drafts
   (`DraftEstimate`). It starts as the sampler's probability. Every 256 drafts verified it takes one Newton step
   on them, damped so that 500 drafts move it half a step. A verified draft is one the main model accepted, or the
   first it rejected. One estimate serves all slots.
3. After a step of several slots, the rows the next round can spare are shared out before the drafter runs
   (`share_rows`). Each writing slot gets one draft. The rest go one at a time to the slot whose next draft is
   the likeliest to be accepted, by the slot's mean acceptance a draft over its last rounds. Each slot then drafts
   one more than item 4 would verify of its share by that mean acceptance (`share_drafts`), so the drafter does not
   make drafts the round will not verify. It drafts the whole share while fewer than two widths are timed.
4. Before a round of several slots, the drafts of all its slots are ranked by the chance that the draft and the
   slot's drafts before it are all accepted. The round verifies the N likeliest, where N gives the most expected
   tokens a millisecond by the measured window times (`choose_verified`). It verifies every draft while fewer than
   two widths are timed.
5. Every eighth time item 4 leaves no draft at all, the round verifies each slot's first draft.
6. A slot makes no more drafts than the tokens it may still write after its next one (`max_tokens`).
7. A lone slot's round is as before: every draft above `draft_min_p`, or the lookup's
   ([ADR-0041](0041-drafts-from-the-slots-own-tokens.md)). `draft_min_p` still ends a slot's drafts in every
   round.

## Why

No part of this is measured on the cards. The gain expected is where slots differ: a slot on text the drafter
predicts well takes rows that an equal share gave to a slot whose drafts are rejected. The constants in item 2 are
*estimates*: 256 and the ridge of 1e-3 follow vLLM's, and the damping of 500 is the value at which a host test with
made-up acceptances settles within 0.05 of them (`draft_choice_test`).

Item 3 exists because the block pass is bounded (Context): the rows must be shared before the drafts' probabilities
are known, so it uses what each slot accepted lately. Item 5 keeps the estimate and the window times measured: a
round with no drafts measures neither, and one low value would otherwise hold the choice at none.

## Options not chosen

- The same rule for a lone slot. Its drafts are measured as they are
  ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-on-the-cards)), and a rule that verifies fewer
  of them needs its own measurement.
- A block pass of 64 rows, so that every slot is drafted 7 tokens and item 3 goes. The drafter's kernels and lists
  hold 16 rows.
- Every width timed at load. A step at load has no context and no request's routing, so its times are lower than
  a request's; a width the chooser has not run is priced on the line through the two widest it has run.
- An estimate for each slot. A slot's rounds are too few to fit 8 numbers; item 3 uses the slot's own acceptance.
- The drafter's context pass skipped for a slot that has finished. It saves one pass a request, and whether a later
  request on the slot reads those cells is not checked.

## Consequences

- `choose_shared_drafts` and its fixed 2 drafts until a window is timed are gone. The first round of several slots
  verifies every draft it holds.
- Slots in one round may verify different numbers of drafts, from none to 7.
- The estimate learns only from drafts that were verified, and the draft layer of either kind (DFlash 2 or the MTP
  layer) feeds it.
- The engine reports nothing about the estimate. Its slope and offsets are not in `engine_stats`.
- `draft_choice_test` checks items 2 to 4 without a card. The change is not accepted until the alternating runs in
  "Revisit when" show a gain.
