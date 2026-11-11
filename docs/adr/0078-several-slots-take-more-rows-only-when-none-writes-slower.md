# ADR-0078: several slots take more rows only when none writes slower

**Status:** draft, 2026-11-11; not accepted.
**Revisit when:** two, three or four requests writing at once, this change against the commit before it in
alternating order, give any request fewer tokens a second, or fewer in total.

## Context

- Before [ADR-0077](0077-a-decode-step-holds-16-rows.md) a round of several slots shared a step of 8 rows. Every
  slot made the same number of drafts, at most its part of the spare rows: the number with the most tokens a
  millisecond, by each slot's acceptance and the time of windows of each width (`choose_shared_drafts`,
  [the record](../measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers)).
- A step now holds 16 rows. Choosing the number over 16 rows the same way, and ranking the drafts of all slots by
  their chance of being accepted, both gave fewer tokens a second than the 8-row choice, 1 to 12% in alternating
  runs on two cards (*estimate*: from runs with no record in MEASUREMENTS.md). Ranking gives the rows to the slots
  whose drafts are accepted, so a slot whose drafts are rejected writes slower.
- On two cards a window's time is about a fixed time for its width (22 ms for 3 rows, 32 for 8, 40 for 9, 48 for
  16) and 0.026 ms for each expert the CPU computes in it. The experts a row misses change with the requests' text:
  about 8 a row with four requests, about 100 with two (*estimate*, from the same runs). One table of times by
  width mixes requests with different misses, and a width timed once keeps that time.
- After a round, the draft layer's pass takes 4 to 5 ms for a block of up to 8 rows and about 6.5 ms past 8
  (*estimate*, from the same runs). A round pays it whatever its drafts.

## Decision

1. The core times each step and round of several slots in two parts: its window, by its rows and the experts the
   CPU computed in it (`RoundTimes`), and what follows up to the draft layer's drafts, by the window's rows
   (`drafter_times`). `RoundTimes` fits one cost a miss across the widths and keeps each width's time less that
   cost of its misses, and the misses a row of the rounds lately. A width's time is that net time and its rows'
   misses at the cost. A width not timed twice is on the line through the nearest timed ones; below the
   narrowest, on the line through the two narrowest, and no less than in proportion to the rows.
2. The even split is the choice before ADR-0077: one number for all slots, each at most its part of a step of 8
   rows, the number with the most tokens a millisecond of the window (`choose_shared_drafts`).
3. A number of the whole step's 16 rows, shared out the same way, replaces the even split when it gives more tokens
   a millisecond in total and every slot at least its tokens a millisecond in the even split, both priced with the
   window and the draft layer's pass. Of those, the one with the most tokens a millisecond.
4. Until two widths are timed: 2 drafts each of 8 rows, and once that width is timed, 3 each of 16 rows, so that a
   wider one is timed too. Every eighth choice is 2 each of 8 rows, as before ADR-0077.
5. A round verifies every draft made. A lone slot's round is unchanged.

## Why

Item 3 keeps every request at least as fast as the 8-row choice, by the core's own estimates, and gives the step's
other rows only where all the requests gain. A request's speed beside others no longer depends on whether their
drafts are accepted.

A host simulation runs this code and the 8-row choice on made-up requests: acceptances that fall 0.05 a place, each
request's rows missing at its own rate, and the costs in Context with 1.5 ms between rounds. In nine cases of 2 to 6
requests, 3,000 rounds each, eight are within 0.2% of the 8-row choice in total and within 0.6% for the slowest
request; four requests with 8 misses a row give 7.4% more in total and 7.3% more for the slowest. This is not a
measurement of the cards.

Item 2 prices the even split by the window alone, as before. With the draft layer's pass, more drafts look cheaper,
and one acceptance a slot for all places overstates its later drafts. In the simulation that chose more drafts
where acceptance is 0.5 to 0.6, and the slowest request lost 7%.

Item 1's line below the narrowest width: a width priced as the narrowest timed one never looks cheaper, so it is
never run. In the simulation, three requests that started at 8 and 12 rows never ran 6 and stayed at 8, 1.7% slower
in total. Item 4 times two widths because a step with no drafts, one row a slot, is too rare to be timed twice.

## Options not chosen

- Ranking the drafts of all slots by their chance of being accepted, as vLLM's adaptive verification does
  (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` at commit `b2b5ae8`). It was 1 to 12% under the 8-row
  choice on the cards (Context).
- A number of drafts for each slot, the most tokens a millisecond with the floor of item 3. Each slot's acceptance
  rests on its own few rounds; in the simulation four requests with few misses were 4.7% slower than the 8-row
  choice.
- An acceptance for each place in a slot's drafts. A later place is checked only when the ones before it are
  accepted, so its rate rests on few rounds.
- Every width timed at load. A step at load has no request's routing, so its misses are not a request's.

## Consequences

- `choose_shared_drafts` gives each slot's drafts. `window_times` still times a lone slot's rounds for the lookup's
  choice.
- With four requests whose rows miss few experts, rounds run 16 rows, 3 drafts each, where they ran 8, 1 each.
- The first rounds of several slots run 2 drafts each and then 3, before the times decide.
- `lookup_test` checks items 1 to 4 without a card. The change is not accepted until the alternating runs in
  "Revisit when" show no request slower.
