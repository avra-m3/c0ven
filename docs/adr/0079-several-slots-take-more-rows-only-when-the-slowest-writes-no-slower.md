# ADR-0079: several slots take more rows only when the slowest writes no slower

**Status:** draft, 2026-11-11; not accepted.
**Revisit when:** two, three or four requests writing at once, this change against the commit before it in
alternating order, give the slowest request fewer tokens a second, or fewer in total.

## Context

- Before [ADR-0078](0078-a-decode-step-holds-16-rows.md) a round of several slots shared a step of 8 rows. Every
  slot made the same number of drafts, at most its part of the spare rows: the number with the most tokens a
  millisecond, by each slot's acceptance and the time of windows of each width (`choose_shared_drafts`,
  [the record](../measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers)).
- A step now holds 16 rows. Choosing the number over 16 rows the same way, and ranking the drafts of all slots by
  their chance of being accepted, both gave fewer tokens a second than the 8-row choice, 1 to 12% in alternating
  runs on two cards (*estimate*: from runs with no record in MEASUREMENTS.md). Ranking gives the rows to the slots
  whose drafts are accepted, so a slot whose drafts are rejected writes slower.
- On two cards a window's time is about a fixed time for its width (22.2 ms for 3 rows, 32.2 for 8, 48.1 for 16)
  and 0.026 ms for each expert the CPU computes in it; a ninth row adds about 3 ms to a window of 8
  ([the record](../measurements/MEASUREMENTS.md#2026-11-11-products-read-the-steps-rows)). The experts a row misses
  change with the requests' text: about 8 a row with four requests, about 100 with two. After a round, the draft
  layer's pass takes 4.8 to 5.1 ms for a block of 4 to 8 rows and 7.0 to 7.4 ms past 8
  ([the record](../measurements/MEASUREMENTS.md#2026-11-11-step-rows-16-one-draft-count)); a round pays it whatever
  its drafts. One table of times by width mixes requests with different misses, and a width timed once keeps that
  time.
- The 8-row choice prices a width by the mean of all windows of that width since load, of any number of slots and
  any text. With four requests, after three requests of mixed text, it priced 8 rows at 46.8 ms where the four
  requests' rounds took 32 to 34, and 4 rows from lone rounds; it gave the four requests no drafts. A step without
  drafts does not time a width, and measures no acceptance, so a request stayed at its first guess of 2 in 3 and
  without drafts but for every eighth choice. In alternating runs on two cards, runs of the 8-row choice and of
  this change alike went 59 to 79 steps without drafts with four requests, 3 to 5% under the other runs, and up to
  177 steps with four requests editing text, 9% under (*estimate*, from runs with no record in MEASUREMENTS.md).
- A slot's drafts are accepted less often the later their place, and the draft layer stops at a draft below its
  probability floor. In a log of every round on two cards, three requests on prose made 0.81, 0.73 and 0.53 of the
  first three drafts asked for, and had 0.83, 0.74 and 0.65 of those made accepted; on prose a round held fewer
  drafts than it asked for in 33 to 46% of its slots, on copied and edited text in 8 to 15%
  ([the record](../measurements/MEASUREMENTS.md#2026-11-11-step-rows-16-one-draft-count)). Counting a draft asked
  for and not made as rejected, prose had 0.59 to 0.75 at the first place and 0.43 to 0.68 at the second; copied and
  edited text 0.98 to 1.00 at the first and 0.81 to 1.00 later (*estimate*, from the same log).
- In the same log, by the drafts asked: four requests on prose wrote 0.172 tokens a ms with 1 each and 0.171 with
  2 each; three on prose 0.151 with 1, 1 and 1 and 0.152 with 2, 2 and 1; three of mixed text 0.129 and 0.119;
  three copying text 0.239 with 4 each of 15 rows against 0.196; four editing text 0.171 with 3 each of 16 rows
  against 0.150 with 1 each (*estimate*).
- A first version of this change counted a draft asked for and not made as rejected at its place, and priced a
  round at the rows asked for. With three requests it ran 5.8 rows a round where the 8-row choice ran 6.1, and was
  1.5% slower on prose, 2.6% on mixed text and 6.0% on mixed text sampled at temperature 1; on mixed text the slowest
  request wrote 33.4 tokens a second against 35.9. With four requests it was level, on copied text 12% faster and on
  edited text 14% (*estimate*, from alternating runs on two cards with no record in MEASUREMENTS.md).

## Decision

1. The core times each step and round of several slots in two parts: its window, by its rows and the experts the
   CPU computed in it (`RoundTimes`), and what follows up to the draft layer's drafts, by the window's rows
   (`drafter_times`). `RoundTimes` fits one cost a miss across the widths and keeps each width's time less that
   cost of its misses, and the misses a row of the rounds lately. A width's time is that net time and its rows'
   misses at the cost. A width not timed twice, there and in `drafter_times` alike, is on the line through the
   nearest timed ones; past the widest, from the widest at the gentler slope of the line through the two widest and
   the line through the narrowest and the widest, and no less than the widest; below the narrowest, on the line
   through the two narrowest, no dearer than the narrowest and no less than in proportion to the rows. A round of a
   number of drafts is priced at the time of each width it may have and the draft layer's pass after it, by the
   chance of that width from item 2's shares of drafts made.
2. Each slot keeps two figures at each of the first 8 places of its drafts (`DraftOdds`): the share of the rounds
   that asked for a draft there, after the draft layer made the one before, in which it made it; and the acceptance
   of the drafts made there, in the rounds that checked the place, every draft before it accepted. Each is decayed
   by 0.9 a round that tries or checks the place, and for the weight those rounds do not yet carry takes the place
   before it's (the first place's: all made, and the slot's acceptance a draft). A number of drafts asked for is
   worth the row sampled last and, for each place, the product of the places' shares made and acceptances up to
   it; it makes each number of drafts up to it with the chance the shares made give. A lone slot's choice keeps one
   acceptance for all places.
3. The even split is one number for all slots, each at most its part of a step of 8 rows: the number with the most
   tokens a millisecond by items 1 and 2; 2 each while fewer than two widths are timed and every eighth choice; no
   drafts only once each slot's first draft has been tried or checked in 4 rounds.
4. The wide count is a number of the whole step's 16 rows, shared out the same way, that gives more tokens a
   millisecond in total, and its slowest slot at least the tokens a millisecond of the even split's slowest, both
   by items 1 and 2; of those, the one with the most. Until two widths are timed by `RoundTimes`, 3 each of 16 rows,
   so that a wider one is timed. Only a number whose next round the step could hold counts (ADR-0078). Where none
   qualifies, the one of those with the most tokens a millisecond is the wide count for item 5's checks alone, so
   that a wide width priced too high is timed again. Where the step cannot hold a round's rows when it runs, as the
   ring shrinks between the choice and the round, the round takes 8 rows, shared out one draft at a time to each
   slot with more drafts made, and counts in neither arm of item 5's trial.
5. Where there is a wide count, a trial for each number of slots (`RoundTrial`) chooses between it and the even
   split. Each arm keeps the tokens its rounds wrote over the time of their windows and of the draft layer's passes
   after them, and items 3 and 4's estimates for those rounds, each decayed by 1/16 a round of its own; its
   correction is the first rate over the second, starting from 0.85 for the wide count and 1 for the even split, as
   4 rounds at the arm's first estimates. A round takes the arm whose estimate, times its correction, is the higher
   (the even split where the wide count does not qualify), and every 32nd choice the other (every 16th while the
   other has fewer than 8 rounds), so that its correction and its widths' times stay current. A choice shares
   drafts out among the slots that will draft, not those that finished or still read their prompt. A round counts in
   an arm only where each of its slots holds the drafts that arm's last choice gave it: not where a slot's drafts
   came from elsewhere since (a lone slot's round, a prompt's end) or a slot joined or left.
6. A round verifies every draft made. A lone slot's round is unchanged.
7. Where a number leaves some slots one draft more than others (2, 2 and 1 of 8 rows for three slots), those drafts
   go first to the slots one more draft makes faster (its tokens over the round's time with its row, by items 1 and
   2, above its tokens without it), and of those to the slots expected to write the fewest tokens. Two slots next
   to each other in that order are close where one more draft would take the first past the second. Where the line
   between the slots that take one more and those that do not falls among close slots, those slots take the drafts
   left in turns, a choice at a time. Items 3 and 4 take each slot's tokens and the round's time as their means over
   the turns.

## Why

Item 1 prices both choices with the misses of the rounds lately and the times of rounds of several slots, steps
without drafts among them, so the even split follows the requests that run now. A request starts at the acceptance
of 2 in 3 a draft, and a step without drafts measures nothing; item 3 gives it drafts until its first place is
measured, so a low first guess cannot keep it without drafts.

Item 2 prices each draft at its own place. With one acceptance a slot, a prose request's first drafts, most of what
it checks, set the price of its third, and a count of 3 each looked 11% faster than 1 each for four requests on
prose where the round log has them level. A later place is checked only when the ones before it are accepted, so its
rounds are few; a place takes the place before it's for the weight its rounds do not carry, and keeps what its
rounds gave while no round reaches it. A draft asked for and not made gives no token and takes no row. Counted as
rejected at the rows asked for, it priced such rounds 4 to 8% high in the round log, and three requests ran fewer
rows than the 8-row choice and wrote slower (Context). The share made prices both the tokens and the rows. With
the shares in Context for three requests on prose and `lookup_test`'s times, 2, 2 and 1 drafts are 5% faster than 1
each, and priced at the rows asked 1 each was 1% faster. Each width is priced by its chance because a window's time
jumps from 8 rows to 9 (Context): with the same figures, 2 each of 16 rows looks 1.9% faster than 2, 2 and 1 at the mean
of the rows made, and is 2.3% slower by each width's chance. On copied and edited text the draft layer makes nearly
every draft asked, so there the count of 16 rows keeps its gain.

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

Item 7 gives the spare drafts to the requests that write slowest, where a draft helps them, and keeps a slot's
place in the round from choosing which request writes slower. With the spare drafts always at the first slots, as
with steps of 8 rows, three requests of one kind wrote at the slowest 0.890 of the fastest's rate copying text and
0.904 on prose sampled; with them taking turns between all slots, 0.991 and 0.955, in alternating runs on two cards
([the record](../measurements/MEASUREMENTS.md#2026-11-12-spare-drafts-take-turns)). A request's slot depends on the
order the requests came in and on which slots were freed last, so with the spare drafts at the first slots the slow
request of three of mixed text has them in some orders and not in others. With the requests sent in three orders,
taking turns between all slots left the slow request 0.5 tokens a second under main's rate, behind in all four
rounds; given first to the slowest, as item 7 does, it wrote 0.6 more than on main, ahead in all three
([the record](../measurements/MEASUREMENTS.md#2026-11-12-spare-drafts-to-the-slowest)). In a trace of every round,
the request that writes the fewest tokens a round was the one given a draft less in 1 to 6 of 28 to 87 rounds of
mixed text, and each of three requests of one kind in 26% to 49% of the rounds. Requests of one kind differ by
less than a draft's worth, so close slots take turns; without turns, the one a little behind would take every
spare draft and write faster than the others. A slot that one more draft makes slower takes none before the others:
at a fifth draft, the slow request of mixed text is expected to accept it about 1 time in 100, and its row would
make every slot's round longer (*estimate*, from logged choices). Item 3's number is also the one with the largest
sum of the slots' log rates, which a gain for one slot bought by a larger loss for another lowers, in all but 1 to
4 of 44 to 137 logged choices a case (*estimate*, from two logged loads).

With a floor for each slot instead of the slowest, a host simulation ran item 4 alone and the 8-row choice on
made-up requests: acceptances that fall 0.05 a place, each request's rows missing at its own rate, and the costs in
Context with 1.5 ms between rounds. In nine cases of 2 to 6 requests, 3,000 rounds each, eight are within 0.2% of
the 8-row choice in total and within 0.6% for the slowest request; four requests with 8 misses a row give 7.4% more
in total and 7.3% more for the slowest. This is not a measurement of the cards.

Item 1's line below the narrowest width: a width priced as the narrowest timed one never looks cheaper, so it is
never run. In the simulation, three requests that started at 8 and 12 rows never ran 6 and stayed at 8, 1.7% slower
in total. Item 4's first wide count times a second width because a step with no drafts, one row a slot, is too
rare to be timed twice.

Item 1's slope past the widest width and item 4's count for the checks: the same holds for wide widths priced too
high. In a trace of three requests copying text on two cards, two slow rounds at 11 and 12 rows put the line
through the two widest at 14 ms a row and 16 rows at 162 ms. No wide count qualified after that, so none ran and
those widths were not timed again: the requests took 8 rows or fewer in 104 of 111 rounds and wrote 124.0 tokens a
second, the slowest 41.1, where three other loads of the same build or the one before timed 16 rows and wrote
131.5 to 134.7, the slowest 42.7 to 44.8. In two of those, the line through 3 and 12 rows put 16 at 124.5 and
110.0 ms against the 126.2 and 111.0 timed, and the line through 11 and 12 at 140.3 and 114.2. With this item 1
and item 4, two loads ran wider than 8 rows in 50 of 86 and 39 of 77 rounds of copying text
([the record](../measurements/MEASUREMENTS.md#2026-11-12-wide-widths-timed-again)). A width priced too low is run
and timed; the checks cost a wide count that does not qualify one round in 32 once it has 8.

## Options not chosen

- Ranking the drafts of all slots by their chance of being accepted, as vLLM's adaptive verification does
  (`vllm/v1/worker/gpu/spec_decode/adaptive_verification.py` at commit `b2b5ae8`). It was 1 to 12% under the 8-row
  choice on the cards (Context).
- Item 4's estimate alone, without the trial, with one acceptance a slot for all places. It was 2 to 6% under the
  8-row choice with three and four requests (*estimate*, from alternating runs on two cards).
- The even split priced as before ADR-0078, by the mean window of each width since load. It mixes numbers of
  slots and texts, and a request it gives no drafts stays without them (Context).
- Counting a draft asked for and not made as rejected, and pricing a round at the rows asked for. It leans to fewer
  rows, and three requests wrote 1.5 to 6% slower than with the 8-row choice (Context).
- Pricing a round at the mean of the rows made. It hides the jump in a window's time past 8 rows (Why).
- A trial of every number of drafts. Each needs its own rounds, and a slow one costs every request while it runs.
- A trial that compares the arms' measured rates. One arm's rate is from whatever text it last ran on (Why).
- A number of drafts for each slot, the most tokens a millisecond with the floor of item 4. Each slot's acceptance
  rests on its own few rounds; in the simulation four requests with few misses were 4.7% slower than the 8-row
  choice.
- No slot with fewer tokens a millisecond than in the even split. Its spare drafts go to some slots and not others,
  so the wide count had to beat those slots, not the slowest (Why).
- The spare drafts always at the first slots, as before item 7 and with steps of 8 rows. The order the requests
  came in then picks which of them writes slower (Why).
- The spare drafts to the slots most likely to accept them. Over the logged choices of two loads, rebuilt with the
  core's estimates: 0.2% to 2.8% more in total and the slowest slot 2.2 to 14.5 tokens a second less; with three
  requests copying text, which accept nearly every draft, the same slot is always short (*estimate*).
- The spare drafts to the slots expected to write the fewest tokens, with no turns, and to slots they make slower.
  The same estimates: with three requests copying text one slot is always short, and a slot that accepts few
  drafts takes rows it gains nothing from (*estimate*).
- Of the slots one more draft makes faster, the one expected to write the fewest tokens first, then the others by
  what one more adds. With three of mixed text on two cards, 2.9% more in total than main and the slow request 0.7
  tokens a second faster; sampled, 6.4% more in total and the slow request 2.7 slower, in every order of two rounds.
  The request that writes the most takes a spare draft in most rounds and makes nearly every draft it asks for, so
  each round is longer, while the other two share one spare draft
  ([the record](../measurements/MEASUREMENTS.md#2026-11-12-spare-drafts-to-the-slowest)).
- The spare drafts to the next slots at each choice, whatever their tokens (item 7 before this). With three of
  mixed text sent in three orders, the slow request wrote 0.5 tokens a second less than on main and the slowest
  over the fastest was 0.044 lower; with item 7, 0.6 more and 0.009 higher (Why).
- A number taken only where the slowest slot writes no slower than with one draft fewer each. A trace found the
  core's estimates within 3% of the rounds' tokens a millisecond on mixed text at the number it takes most. By them,
  on the logged choices of two loads: 4.8% to 6.8% less in total with two requests and 5.8% to 10.0% less with three
  of mixed text, for the slowest 3.4% to 8.5% more (*estimate*).
- The number of drafts that gives the slowest slot the most. The same estimates: 4.1% to 10.4% less in total with
  two requests and in mixed text, for the slowest slot 1.8% to 6.7% more (*estimate*).
- Every width timed at load. A step at load has no request's routing, so its misses are not a request's.

## Consequences

- `choose_shared_drafts` gives the even split and the wide count; the core keeps one `RoundTrial` for each number
  of slots, for as long as the engine runs. A trial's rates come from earlier requests too; the decay and the
  sixteenth choice let new ones replace them.
- Several slots' choices differ from the 8-row choice before ADR-0078 even within 8 rows: they follow the misses of
  the rounds lately and each place's acceptance and share made.
- Four requests no longer run tens of steps without drafts at the start of their text.
- A number of slots whose wide count item 4 estimates at least 18% faster runs it from its first round.
- Requests that repeat text they were given, such as copies or code returned with a change, accept nearly every
  draft; there the wide count's rounds write the most. On prose, later drafts are accepted less often and often not
  made, and the wide count is rarely offered.
- With three requests that accept most drafts, a round can run 4 drafts each of 16 rows where the even split ran
  2, 2 and 1 of 8; the first two slots may write slightly fewer tokens a millisecond than in the even split.
- In eight rounds of alternating runs on two cards against steps of 8 rows, with the spare drafts taking turns
  between all slots (item 7 before this), four editing code wrote 11.2% more tokens a second in total, the slowest
  34.8 → 39.7; one, two and four requests -2.4% to +0.2%, each within its spread, the slowest within 0.4. One and two
  requests differ in the swap rounds alone, which wait for the swap thread's reads of the model file longer in one
  arm or the other from run to run. Item 7 changes only rounds whose spare drafts do not share out evenly, here
  those of three requests. With item 7, in three rounds, the mixed text sent in three orders: three on prose +0.6%,
  sampled +1.8%, copying text +2.4% (+5.8% in the eight rounds before), mixed text +0.7% and sampled +0.2%. The
  slowest request: three on prose sampled 49.8 → 51.3 tokens a second, copying 63.1 → 66.8, three on prose level;
  mixed text 35.3 → 35.9, ahead in all three rounds, and sampled 42.3 → 40.6, behind in all three. Sampled, the slow
  request's answer ends on its own and its rate rises about 1.8 tokens a second for each 100 tokens it writes; its
  answers ran shorter in the PR's runs (median 149 tokens against 200, 153 in both arms in the series before), and at
  the same length it wrote level with main (+0.2 ± 0.8 tokens a second).
- On one card, with the spare drafts taking turns, in three rounds, editing code wrote 9.8% more and the other cases
  -2.2% to +6.2%, each within its spread. With item 7, in three rounds with the three requests sent together, over the
  time all three write at once: copying text 16.7% more in total, its slowest 15.9 → 20.3 tokens a second; mixed text
  +1.2%, its slowest 15.6 → 16.2, both ahead in all three rounds; mixed text sampled -0.3%, its slowest 19.1 → 19.4;
  three on prose +3.7% and sampled +1.6%, each within its spread. Over whole answers mixed text is 0.1% lower in
  total, behind in all three rounds
  ([the record](../measurements/MEASUREMENTS.md#2026-11-12-spare-drafts-to-the-slowest)).
- On two cards, with item 7, in three rounds with the three requests sent together, measured the same way: the
  slowest request faster in all three rounds on prose (+1.6 tokens a second), prose sampled (+3.2) and copying text
  (49.5 → 65.4); the totals prose +0.4%, sampled +3.0%, copying +4.8%. Mixed text in five rounds: -0.9% ± 2.7 in
  total, its slowest +0.9, faster in all five; sampled +2.6% ± 2.0, its slowest +0.6 ± 1.1. In one order of mixed
  text the slow request gains 4.0 tokens a second and the easy one loses about 9.
- `lookup_test` checks items 1 to 5 and 7 without a card. The change is not accepted until the alternating runs in
  "Revisit when" show the slowest request no slower and the total no lower. With the requests sent together, on one
  card and on two, no case's slowest or total is lower beyond its spread. With the requests 20 ms apart on two cards,
  three requests of mixed text sampled are slower, level at the same length.
