# ADR-0077: a decode step holds 16 rows, and one slot's window 8

**Status:** draft, 2026-11-10; not accepted. Supersedes the window size in
[ADR-0050](0050-dflash-2-drafts.md) ("a step's window holds up to 8 rows") for steps of several slots.
**Revisit when:** three requests writing at once, this change against the commit before it in alternating order,
give no more tokens a second in total, or the same-tokens check differs.

## Context

- A draft round runs every writing slot's window in one step, and the slots share the step's rows
  ([DESIGN.md](../DESIGN.md#drafts)). With 8 rows a step, two slots take up to 3 drafts each, three take 2, 2 and
  1, four take 1 each, and eight take none.
- DFlash 2 drafts a block of up to 7 tokens in one pass. Scored offline its third draft is accepted 0.732 of the
  time and its 4th to 7th 0.670 to 0.525
  ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-acceptance)), so with several slots the step's
  rows run out before the drafts do.
- The core chooses how many drafts several slots make from their acceptance and the measured time of windows of
  each width (`choose_shared_drafts`, [the record](../measurements/MEASUREMENTS.md#2026-10-19-draft-rounds-several-writers)).
  It cannot choose more than the step holds.
- A round's window on two cards takes a fixed time for its width, 32.2 ms at 8 rows and 48.1 at 16, and 0.026 ms
  for each expert the CPU computes, about 100 a row with two requests and 8 with four
  ([the record](../measurements/MEASUREMENTS.md#2026-11-11-step-rows-16-one-draft-count)).
- A step runs the list recorded for the next width up. Where the dense products computed every row of that width
  and the XMX products' block reads fetched the rest of the last tile of 8 rows, three slots' window of 9 rows took
  40.9 ms against 33.3 at 8; computing only the step's rows, 36.6
  ([the record](../measurements/MEASUREMENTS.md#2026-11-11-products-read-the-steps-rows)).

## Decision

1. A decode step holds up to 16 rows of all its slots together (`STEP_MAX_ROWS`). Its widths are 1, 2, 3, 4, 6, 8,
   12 and 16.
2. One slot's rows in a step are at most 8, its token and up to 7 drafts (`STEP_MAX_RUN`). DFlash 2's block, a
   lone slot's window, a writer's run in a prompt chunk, the prompt's last rows the drafter starts from and a
   short prompt piece's decode windows stay 8 rows.
3. A group of a cached expert's assignments holds at most 8 (`MOE_GROUP_MAX`). An expert that more than 8 rows
   route to takes one group for each 8 of its assignments, in order. A step of more than 8 rows runs the expert
   kernels built for groups of 8.
4. The BF16 products on the XMX units take a step's rows in tiles of 8, a second tile past 8 rows, and their block
   reads stop at the step's last row. Rows 0 to 7 have the same bits as with one tile.
5. `engine_info.max_step_rows` reports 16. `ENGINE_MAX_STEP_SLOTS` stays 8, and the ABI version does not change.
6. How many drafts several slots make is chosen as
   [ADR-0078](0078-several-slots-take-more-rows-only-when-the-slowest-writes-no-slower.md) says. The step's rows let two
   slots make up to 7 each, three up to 5, 4 and 4, four up to 3 each, and eight 1 each.
7. A step of up to 8 rows runs step kernels built for 8 rows (`STEP_NARROW_ROWS`; the modules `common_r8`,
   `qwen4exp_r8` and `qwen4exp_e256_r8`, built with `-DSTEP_ROWS=8`): the registers, tiles and grouping work-group
   of 128 they had before this change. A wider step runs the modules built for 16 rows.
8. With the streamed pool (`kv_resident`), the ring's fixed part holds the claims of a step of 8 rows, as before,
   and `kv_resident`'s least value is the same. A step of more than 8 rows runs only where the ring holds every page
   its claim can take (`Model::step_fits`): each slot's rows take at most `kvs_row_claims` pages a row, and no page
   past its last row's and the 32 after it. Where it does not, the round takes 8 rows, shared out evenly.
9. The dense products (`matvec`, `matvec_ks`) compute only the step's rows: they read the count from the step plan,
   as the other step kernels do, and the list's width only picks their module.
10. The indexer's scores, the attention's parts and the head's logits share one buffer, since a step uses each in
    turn: the scores until a layer's candidates are taken, the parts in that layer's attention after them, the logits
    after the last layer. The scores buffer holds the card's own blocks, the rows its kernels read. A step of 16 rows
    then leaves the expert cache as much room as one of 8 did.

## Why

A step of more than 8 rows pays where the slots accept most of their drafts. In alternating runs on two cards, with
the drafts chosen as ADR-0078 says, three requests copying text wrote 2.4% to 5.8% more tokens a second than with
steps of 8 rows and four editing code 11.2% more. One, two and four requests wrote -2.4% to +0.2%, each within its
spread, and three on prose, on prose sampled and of mixed text +0.2% to +1.8%. On one card, in three rounds, four
editing code wrote 9.8% more and three copying text 15.8% more, the other cases -1.4% to +6.2%, each within its
spread ([the record](../measurements/MEASUREMENTS.md#2026-11-12-spare-drafts-to-the-slowest)). A window's ninth row
still costs more than its eighth (Context), and on prose later drafts are accepted less often and often not made, so
there a step of more than 8 rows is chosen less often.

Items 4 and 9 keep a step from paying for rows it does not have. They took 4.2 ms off three slots' window of 9 rows
and 3.6 ms off one of 13, and 0.6 and 0.7 ms off windows of 5 and 7 rows, which run the lists of 6 and 8, with the
same candidates (Context).

Items 7 and 8 keep the steps of 8 rows or fewer as they were. Without them, in alternating runs on two cards, a lone
slot's window took 0.5 to 0.7 ms longer than before at the same width, and a ring sized for 16 rows' claims left the
cache about 3 experts a layer fewer (361 against 364) (*estimate*: from runs with no record in MEASUREMENTS.md). Most
rounds of several slots run 8 rows or fewer.

## Options not chosen

- Groups of up to 16 assignments. The expert kernels keep a group's sums in registers; 16 would need new modules
  for every type and spill on the gate and up kernels. Few experts are picked by more than 8 of 16 rows.
- A block of 16 for DFlash 2. Its checkpoint was trained with blocks of 8, and the load refuses another size.
- A 16-row step for a lone slot. It takes 7 drafts already, the most one block gives.

## Consequences

- A load records the decode step at two more widths for each card, once or twice each as the other widths are, and
  the drafter's lists at width 16: two for DFlash 2, seven for the MTP layer. The build has 60 more modules: 57
  `matvec_<type>_l<lanes>_t16`, which the dense products of a step of more than 8 rows use, and the three of item 7.
  Load time and card memory with them are not measured.
- The decode step's buffers hold 16 rows where they held 8, and `StepPlan`, `DraftPlan` and `CommitPlan` are twice
  as long. With item 10, a load on two cards has 17 spare blocks of the expert cache against 16 with 8 rows, and
  caches 364.0 experts a layer after its first refill, as with 8 rows; without it, 12 spare blocks and 363.8
  ([the record](../measurements/MEASUREMENTS.md#2026-11-11-products-read-the-steps-rows)).
- In a step of more than 8 rows, `matvec_bf16` and `matvec_bf16_ks` keep 16 sums a lane.
- With long contexts and a small ring, several slots draft over 8 rows only, as before this change (item 8).
- Steps of two, four and more slots may also draft more than before (item 6), so a check of this change covers
  those counts as well as three.
- `matvec_test` gives the dense products their row count in memory, as a step does (item 9).
- `step_rows_test` checks items 3 and 4 on a card. The change is not accepted until it has passed there, with
  `api_test` and the same-tokens check, and the alternating runs in "Revisit when" show a gain.
