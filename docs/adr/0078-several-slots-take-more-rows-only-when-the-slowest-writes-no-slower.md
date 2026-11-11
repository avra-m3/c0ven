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
- The 8-row choice prices a width by the mean of all windows of that width since load, of any number of slots and
  any text. With four requests, after three requests of mixed text, it priced 8 rows at 46.8 ms where the four
  requests' rounds took 32 to 34, and 4 rows from lone rounds; it gave the four requests no drafts. A step without
  drafts does not time a width, and measures no acceptance, so a request stayed at its first guess of 2 in 3 and
  without drafts but for every eighth choice. In alternating runs on two cards, runs of the 8-row choice and of
  this change alike went 59 to 79 steps without drafts with four requests, 3 to 5% under the other runs, and up to
  177 steps with four requests editing text, 9% under (*estimate*, from runs with no record in MEASUREMENTS.md).
- A slot's drafts are accepted less often the later their place. In a log of every round on two cards, prose had
  0.80 to 0.82 of first drafts accepted, 0.67 to 0.75 of second drafts and 0.65 of third. The draft layer also
  stops at a draft below its probability floor: on prose it made fewer drafts than asked for 33 to 46% of slots,
  on copied and edited text 8 to 15%. Counting a draft asked for and not made as rejected, prose had 0.59 to 0.75 at
  the first place and 0.43 to 0.68 at the second; copied and edited text 0.98 to 1.00 at the first and 0.81 to 1.00
  later (*estimate*, from one run with a round log).
- In the same log, by the drafts asked: four requests on prose wrote 0.172 tokens a ms with 1 each and 0.171 with
  2 each; three on prose 0.151 with 1, 1 and 1 and 0.152 with 2, 2 and 1; three of mixed text 0.129 and 0.119;
  three copying text 0.239 with 4 each of 15 rows against 0.196; four editing text 0.171 with 3 each of 16 rows
  against 0.150 with 1 each (*estimate*).

## Decision

1. The core times each step and round of several slots in two parts: its window, by its rows and the experts the
   CPU computed in it (`RoundTimes`), and what follows up to the draft layer's drafts, by the window's rows
   (`drafter_times`). `RoundTimes` fits one cost a miss across the widths and keeps each width's time less that
   cost of its misses, and the misses a row of the rounds lately. A width's time is that net time and its rows'
   misses at the cost. A width not timed twice is on the line through the nearest timed ones; below the
   narrowest, on the line through the two narrowest, and no less than in proportion to the rows. A round of a
   number of drafts is priced at that time of its rows and the draft layer's pass.
2. Each slot keeps the draft layer's acceptance at each of the first 8 places of its drafts (`DraftOdds`): the
   rounds that checked the place, every draft before it accepted, each decayed by 0.9 a round that checks it, and
   for the weight those rounds do not yet carry, the place before it's (the first place's from the slot's acceptance
   a draft). A draft that a round of several slots asked for and the draft layer did not make counts as rejected at
   its place. A number of drafts is worth the row sampled last and, for each place, the product of the places'
   acceptances up to it. A lone slot's choice keeps one acceptance for all places.
3. The even split is one number for all slots, each at most its part of a step of 8 rows: the number with the most
   tokens a millisecond by items 1 and 2; 2 each while fewer than two widths are timed and every eighth choice; no
   drafts only once each slot's first draft has been checked in 4 rounds.
4. The wide count is a number of the whole step's 16 rows, shared out the same way, that gives more tokens a
   millisecond in total, and its slowest slot at least the tokens a millisecond of the even split's slowest, both
   by items 1 and 2; of those, the one with the most. Until two widths are timed by `RoundTimes`, 3 each of 16 rows,
   so that a wider one is timed. There is no wide count when no number qualifies, or when the step could not hold
   the next round's rows (ADR-0077).
5. Where there is a wide count, a trial for each number of slots (`RoundTrial`) chooses between it and the even
   split. Each arm keeps the tokens its rounds wrote over the time of their windows and of the draft layer's passes
   after them, and items 3 and 4's estimates for those rounds, each decayed by 1/16 a round of its own; its
   correction is the first rate over the second, starting from 0.85 for the wide count and 1 for the even split, as
   4 rounds at the arm's first estimates. A round takes the arm whose estimate, times its correction, is the higher,
   and every 32nd choice the other (every 16th while the other has fewer than 8 rounds), so that its correction
   stays current.
6. A round verifies every draft made. A lone slot's round is unchanged.

## Why

Item 1 prices both choices with the misses of the rounds lately and the times of rounds of several slots, steps
without drafts among them, so the even split follows the requests that run now. A request starts at the acceptance
of 2 in 3 a draft, and a step without drafts measures nothing; item 3 gives it drafts until its first place is
measured, so a low first guess cannot keep it without drafts.

Item 2 prices each draft at its own place. With one acceptance a slot, a prose request's first drafts, most of what
it checks, set the price of its third, and a count of 3 each looked 11% faster than 1 each for four requests on
prose where the round log has them level. A later place is checked only when the ones before it are accepted, so its
rounds are few; a place takes the place before it's for the weight its rounds do not carry, and keeps what its
rounds gave while no round reaches it. A draft asked for and not made gives no token and a narrower window: counting
it as rejected prices the tokens right, and pricing the rows asked for prices such rounds 4 to 8% high in the round
log, which leans to fewer rows. On copied and edited text the draft layer makes nearly every draft asked, so there
the count of 16 rows keeps its gain.

A case where the wide count never wins runs the even split but for the trial's rounds of it: one in 16 until it
has 8, then one in 32. A wide count 10% slower costs about 0.3% once it has 8 rounds. Item 5 judges the wide count by
the tokens the rounds write, so a wrong acceptance or time in item 4's estimate no longer picks it.

Item 5 compares each arm with its estimate, not with the other arm's rate. An arm that has not run for a while
last ran on other text: in a probe on two cards, three requests copying text after three writing essays kept the
even split, as the wide count's rate was still the essays' (0.16 tokens a ms against the even split's 0.19 on the
copied text), while its rounds on the copied text gave 0.22 (*estimate*, from one run with a round log). The
estimates follow the text the slots write now; the corrections carry what the estimates get wrong for each arm.

Item 5's starting corrections: in the same probe's round log the wide count's correction ended 0.65 to 0.84 and
the even split's 0.74 to 1.00, the wide count's 0.75 to 1.00 of the even split's in every case (*estimate*). An
arm with no rounds yet compared at 1 against 1 ran 8 rounds of the wide count first wherever item 4 offered one;
with two requests, those rounds wrote 13% fewer tokens a millisecond than the even split's. At 0.85 the wide count
runs from the start only where item 4 estimates it at least 18% faster.

Item 4 offers the step's other rows only where the slowest request gains by the core's estimates, so a slot that
accepts few drafts is not made to wait on the others' longer windows. It compares the slowest slots, not each slot
with itself: the even split of three slots is 2, 2 and 1 drafts, so a count of 4 each had to beat the two slots
given the spare row, while the third, the slowest, gained most. In a probe on two cards with a log of every choice,
three requests copying text (acceptance 0.95) had a wide count in 15 of the 80 choices that could take 16 rows,
and the trial took it each time (*estimate*, from one run).

With a floor for each slot instead of the slowest, a host simulation ran item 4 alone and the 8-row choice on
made-up requests: acceptances that fall 0.05 a place, each request's rows missing at its own rate, and the costs in
Context with 1.5 ms between rounds. In nine cases of 2 to 6 requests, 3,000 rounds each, eight are within 0.2% of
the 8-row choice in total and within 0.6% for the slowest request; four requests with 8 misses a row give 7.4% more
in total and 7.3% more for the slowest. This is not a measurement of the cards.

Item 1's line below the narrowest width: a width priced as the narrowest timed one never looks cheaper, so it is
never run. In the simulation, three requests that started at 8 and 12 rows never ran 6 and stayed at 8, 1.7% slower
in total. Item 4's first wide count times a second width because a step with no drafts, one row a slot, is too
rare to be timed twice.

## Options not chosen

- Ranking the drafts of all slots by their chance of being accepted, as vLLM's adaptive verification does
  (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` at commit `b2b5ae8`). It was 1 to 12% under the 8-row
  choice on the cards (Context).
- Item 4's estimate alone, without the trial, with one acceptance a slot for all places. It was 2 to 6% under the
  8-row choice with three and four requests (*estimate*, from alternating runs on two cards).
- The even split priced as before ADR-0077, by the mean window of each width since load. It mixes numbers of
  slots and texts, and a request it gives no drafts stays without them (Context).
- Pricing a round at the rows the draft layer would make rather than those asked. It needs a second figure for
  each place, the drafts made; the rows asked price such rounds a little high, toward fewer rows (Why).
- A trial of every number of drafts. Each needs its own rounds, and a slow one costs every request while it runs.
- A trial that compares the arms' measured rates. One arm's rate is from whatever text it last ran on (Why).
- A number of drafts for each slot, the most tokens a millisecond with the floor of item 4. Each slot's acceptance
  rests on its own few rounds; in the simulation four requests with few misses were 4.7% slower than the 8-row
  choice.
- No slot with fewer tokens a millisecond than in the even split. Its spare rows go to the first slots, so the wide
  count had to beat them, not the slowest slot (Why).
- Every width timed at load. A step at load has no request's routing, so its misses are not a request's.

## Consequences

- `choose_shared_drafts` gives the even split and the wide count; the core keeps one `RoundTrial` for each number
  of slots, for as long as the engine runs. A trial's rates come from earlier requests too; the decay and the
  sixteenth choice let new ones replace them.
- Several slots' choices differ from the 8-row choice before ADR-0077 even within 8 rows: they follow the misses of
  the rounds lately and each place's acceptance. On prose three requests may run 1 draft each where they ran 2, 2
  and 1, which the round log has level.
- Four requests no longer run tens of steps without drafts at the start of their text.
- A number of slots whose wide count item 4 estimates at least 18% faster runs it from its first round.
- Requests that repeat text they were given, such as copies or code returned with a change, accept nearly every
  draft; there the wide count's rounds write the most. On prose, later drafts are accepted less often and often not
  made, and the wide count is rarely offered.
- With three requests that accept most drafts, a round can run 4 drafts each of 16 rows where the even split ran
  2, 2 and 1 of 8; the first two slots may write slightly fewer tokens a millisecond than in the even split.
- `lookup_test` checks items 1 to 5 without a card. The change is not accepted until the alternating runs in
  "Revisit when" show the slowest request no slower.
