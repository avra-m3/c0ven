# ADR-0078: several slots take more rows only when the slowest writes no slower

**Status:** draft, 2026-11-11; not accepted.
**Revisit when:** two, three or four requests writing at once, this change against the commit before it in
alternating order, give the slowest request fewer tokens a second, or fewer in total.

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
- Choosing by the core's estimates alone, with the 8-row choice priced by the times of item 1 below, ran more rows
  a round than the commit before it with four requests (7.7 against 7.2) and was 2% slower in total; three
  requests of mixed text were 6% slower (*estimate*, from the same kind of runs). One acceptance a slot for every
  place in its drafts overstates the later drafts when most rounds accept one.

## Decision

1. The core times each step and round of several slots in two parts: its window, by its rows and the experts the
   CPU computed in it (`RoundTimes`), and what follows up to the draft layer's drafts, by the window's rows
   (`drafter_times`). `RoundTimes` fits one cost a miss across the widths and keeps each width's time less that
   cost of its misses, and the misses a row of the rounds lately. A width's time is that net time and its rows'
   misses at the cost. A width not timed twice is on the line through the nearest timed ones; below the
   narrowest, on the line through the two narrowest, and no less than in proportion to the rows.
2. The even split is the choice before ADR-0077, unchanged: one number for all slots, each at most its part of a
   step of 8 rows, the number with the most tokens a millisecond of the windows of every round (`window_times`),
   2 each while no window is timed and every eighth choice.
3. The wide count is a number of the whole step's 16 rows, shared out the same way, that gives more tokens a
   millisecond in total, and its slowest slot at least the tokens a millisecond of the even split's slowest, both
   priced with item 1's times and the draft layer's pass; of those, the one with the most. Until two widths are timed by
   `RoundTimes`, 3 each of 16 rows, so that a wider one is timed. There is no wide count when no number qualifies,
   or when the step could not hold the next round's rows (ADR-0077).
4. Where there is a wide count, a trial for each number of slots (`RoundTrial`) chooses between it and the even
   split. Item 3 prices both the same way. Each arm keeps the tokens its rounds wrote over the time of their
   windows and of the draft layer's passes after them, and item 3's estimates for those rounds, each decayed by
   1/16 a round of its own; its correction is the first rate over the second, starting from 0.85 for the wide
   count and 1 for the even split, as 4 rounds at the arm's first estimates. A round takes the arm whose estimate,
   times its correction, is the higher, and every 32nd choice the other (every 16th while the other has fewer than
   8 rounds), so that its correction stays current.
5. A round verifies every draft made. A lone slot's round is unchanged.

## Why

Item 2 makes the even split the commit before's choice, so a case where the wide count never wins runs as before
but for the trial's rounds of it: one in 16 until it has 8, then one in 32. A wide count 10% slower costs about
0.3% once it has 8 rounds. Item 4 judges the wide count by the tokens the rounds write, so a wrong acceptance or time
in item 3's estimate no longer picks it.

Item 4 compares each arm with its estimate, not with the other arm's rate. An arm that has not run for a while
last ran on other text: in a probe on two cards, three requests copying text after three writing essays kept the
even split, as the wide count's rate was still the essays' (0.16 tokens a ms against the even split's 0.19 on the
copied text), while its rounds on the copied text gave 0.22 (*estimate*, from one run with a round log). The
estimates follow the text the slots write now; the corrections carry what the estimates get wrong for each arm.

Item 4's starting corrections: in the same probe's round log the wide count's correction ended 0.65 to 0.84 and
the even split's 0.74 to 1.00, the wide count's 0.75 to 1.00 of the even split's in every case (*estimate*). An
arm with no rounds yet compared at 1 against 1 ran 8 rounds of the wide count first wherever item 3 offered one;
with two requests, those rounds wrote 13% fewer tokens a millisecond than the even split's. At 0.85 the wide count
runs from the start only where item 3 estimates it at least 18% faster.

Item 3 offers the step's other rows only where the slowest request gains by the core's estimates, so a slot that
accepts few drafts is not made to wait on the others' longer windows. It compares the slowest slots, not each slot
with itself: the even split of three slots is 2, 2 and 1 drafts, so a count of 4 each had to beat the two slots
given the spare row, while the third, the slowest, gained most. In a probe on two cards with a log of every choice,
three requests copying text (acceptance 0.95) had a wide count in 15 of the 80 choices that could take 16 rows,
and the trial took it each time (*estimate*, from one run).

With a floor for each slot instead of the slowest, a host simulation ran item 3 alone and the 8-row choice on
made-up requests: acceptances that fall 0.05 a place, each request's
rows missing at its own rate, and the costs in Context with 1.5 ms between rounds. In nine cases of 2 to 6
requests, 3,000 rounds each, eight are within 0.2% of the 8-row choice in total and within 0.6% for the slowest
request; four requests with 8 misses a row give 7.4% more in total and 7.3% more for the slowest. This is not a
measurement of the cards.

Item 2 prices the even split by the window alone, as before. With the draft layer's pass, more drafts look cheaper,
and one acceptance a slot for all places overstates its later drafts. In the simulation that chose more drafts
where acceptance is 0.5 to 0.6, and the slowest request lost 7%.

Item 1's line below the narrowest width: a width priced as the narrowest timed one never looks cheaper, so it is
never run. In the simulation, three requests that started at 8 and 12 rows never ran 6 and stayed at 8, 1.7% slower
in total. Item 3's first wide count times a second width because a step with no drafts, one row a slot, is too
rare to be timed twice.

## Options not chosen

- Ranking the drafts of all slots by their chance of being accepted, as vLLM's adaptive verification does
  (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` at commit `b2b5ae8`). It was 1 to 12% under the 8-row
  choice on the cards (Context).
- Item 3's estimate alone, without the trial, and with the even split priced by item 1's times. It was 2 to 6%
  under the 8-row choice with three and four requests (Context).
- A trial of every number of drafts. Each needs its own rounds, and a slow one costs every request while it runs.
- A trial that compares the arms' measured rates. One arm's rate is from whatever text it last ran on (Why).
- A number of drafts for each slot, the most tokens a millisecond with the floor of item 3. Each slot's acceptance
  rests on its own few rounds; in the simulation four requests with few misses were 4.7% slower than the 8-row
  choice.
- No slot with fewer tokens a millisecond than in the even split. Its spare rows go to the first slots, so the wide
  count had to beat them, not the slowest slot (Why).
- An acceptance for each place in a slot's drafts. A later place is checked only when the ones before it are
  accepted, so its rate rests on few rounds.
- Every width timed at load. A step at load has no request's routing, so its misses are not a request's.

## Consequences

- `choose_shared_drafts` gives the even split and the wide count; the core keeps one `RoundTrial` for each number
  of slots, for as long as the engine runs. A trial's rates come from earlier requests too; the decay and the
  sixteenth choice let new ones replace them.
- A number of slots whose wide count item 3 estimates at least 18% faster runs it from its first round.
- Requests that repeat text they were given, such as copies or code returned with a change, accept nearly every
  draft; there the wide count's rounds write the most.
- With four requests whose rows miss few experts, rounds run 16 rows, 3 drafts each, where they ran 8, 1 each, if
  those rounds write more tokens a millisecond.
- With three requests that accept most drafts, a round can run 4 drafts each of 16 rows where the even split ran
  2, 2 and 1 of 8; the first two slots may write slightly fewer tokens a millisecond than in the even split.
- `lookup_test` checks items 1 to 4 without a card. The change is not accepted until the alternating runs in
  "Revisit when" show the slowest request no slower.
