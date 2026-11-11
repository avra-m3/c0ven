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
- One more row in a step of three slots costs about 4.45 ms, most of it the CPU's products for experts the cards do
  not hold (*estimate*: from runs with no record in MEASUREMENTS.md).

## Decision

1. A decode step holds up to 16 rows of all its slots together (`STEP_MAX_ROWS`). Its widths are 1, 2, 3, 4, 6, 8,
   12 and 16.
2. One slot's rows in a step are at most 8, its token and up to 7 drafts (`STEP_MAX_RUN`). DFlash 2's block, a
   lone slot's window, a writer's run in a prompt chunk, the prompt's last rows the drafter starts from and a
   short prompt piece's decode windows stay 8 rows.
3. A group of a cached expert's assignments holds at most 8 (`MOE_GROUP_MAX`). An expert that more than 8 rows
   route to takes one group for each 8 of its assignments, in order. A step of more than 8 rows runs the expert
   kernels built for groups of 8.
4. The BF16 products on the XMX units take a step's rows in tiles of 8, a second tile past 8 rows. Rows 0 to 7
   have the same bits as with one tile.
5. `engine_info.max_step_rows` reports 16. `ENGINE_MAX_STEP_SLOTS` stays 8, and the ABI version does not change.
6. How many drafts several slots make is chosen as
   [ADR-0078](0078-several-slots-take-more-rows-only-when-none-writes-slower.md) says. The step's rows let two
   slots make up to 7 each, three up to 5, 4 and 4, four up to 3 each, and eight 1 each.
7. A step of up to 8 rows runs step kernels built for 8 rows (`STEP_NARROW_ROWS`; the modules `common_r8`,
   `qwen4exp_r8` and `qwen4exp_e256_r8`, built with `-DSTEP_ROWS=8`): the registers, tiles and grouping work-group
   of 128 they had before this change. A wider step runs the modules built for 16 rows.
8. With the streamed pool (`kv_resident`), the ring's fixed part holds the claims of a step of 8 rows, as before,
   and `kv_resident`'s least value is the same. A step of more than 8 rows runs only where the ring holds every page
   its claim can take (`Model::step_fits`): each slot's rows take at most `kvs_row_claims` pages a row, and no page
   past its last row's and the 32 after it. Where it does not, the round takes 8 rows.

## Why

No part of this is measured on the cards. At three writers the gain in total tokens a second is at most about 2%
(*estimate*: the acceptance of DFlash 2's third to fifth drafts above, against 4.45 ms for each row they add). It
can be negative where the extra rows' experts miss the cards more than the estimate assumes. The chooser reads the
measured window times, so it stops adding drafts once a wider window gives fewer tokens a millisecond; one round in
eight runs 2 drafts each whatever it chose.

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
  as long.
- In a step of more than 8 rows, `matvec_bf16` and `matvec_bf16_ks` keep 16 sums a lane.
- With long contexts and a small ring, several slots draft over 8 rows only, as before this change (item 8).
- Steps of two, four and more slots may also draft more than before (item 6), so a check of this change covers
  those counts as well as three.
- `step_rows_test` checks items 3 and 4 on a card. The change is not accepted until it has passed there, with
  `api_test` and the same-tokens check, and the alternating runs in "Revisit when" show a gain.
