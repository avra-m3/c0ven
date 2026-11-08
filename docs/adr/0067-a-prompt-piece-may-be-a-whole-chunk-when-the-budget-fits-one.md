# ADR-0067: a prompt piece beside writers is a whole chunk when the budget fits one

**Status:** draft, 2026-11-08; not accepted. Supersedes in part item 3 of
[ADR-0044](0044-a-short-prompt-is-read-whole.md), where it says a piece holds never more than 1,024 rows.
**Revisit when:** an ABBA run of a mixed load of writers and long prompts, at the default budget and at 2,000 ms,
shows whole-chunk pieces read no more prompt rows a second than pieces kept to the fetch path, or the fetch path's
line changes so that a 1,021-row piece reads as many rows a second as a whole chunk.

## Context

- Beside writers, a prompt piece holds at most 1,021 rows (`whole_rows` of the core's `fetch_rows`), so that its
  chunk, with up to three writers' rows, fetches only the uncached experts its router picks
  ([ADR-0046](0046-a-short-chunk-fetches-the-experts-its-router-picks.md)). A longer chunk copies every uncached
  expert.
- Pieces forced to sizes of 256 to 3,072 rows beside 1, 2 and 4 writers fit two lines: a chunk of up to 1,024 rows
  takes about 247 ms + 0.31 ms a row, a longer one 316 ms + 0.35 ms a row, with 0.19 and 0.07 ms a row more per 100k
  positions. A writer's longest gap is the longest chunk beside it plus 0-85 ms
  ([measurements](../measurements/MEASUREMENTS.md#2026-11-01-whole-rows-and-stretch)).
- A whole chunk of 4,352 rows takes 1,645-1,680 ms at position 0 (2,590-2,646 rows a second) and 2,748 ms at
  position 239,360 ([measurements](../measurements/MEASUREMENTS.md#2026-10-17-prefill-timings)).
- The rows a second of a chunk, from the two lines (*estimate*: the lines are fitted at positions up to 219k; 239,360
  extends them):

  | Piece | Position 0 | Position 140k | Position 239k |
  | --- | --- | --- | --- |
  | 1,021 rows, fetch path | 1,812 | 1,223 | 993 |
  | 1,025 rows, every expert copied | 1,519 | 1,322 | 1,211 |
  | 1,536 rows | 1,799 | 1,530 | 1,383 |
  | 2,048 rows | 1,983 | 1,660 | 1,488 |
  | 3,072 rows | 2,208 | 1,815 | 1,612 |
  | 4,352 rows, a whole chunk | 2,366 | 1,921 | 1,694 |

  At position 0 a piece past the fetch path reads more rows a second than a 1,021-row piece only past about 1,565
  rows; at 140k, past about 854.
- With the default budget, 500 ms, a piece aims at no more than 1.6 times it, 800 ms, which stops it near 1,500 rows
  even without the cap.
- Writers' next rows ride inside a prompt chunk for 5.6-9.8 ms
  ([ADR-0053](0053-writers-rows-ride-in-a-prompt-chunk.md)), so they advance during a long chunk too.
- On a load where one slot is almost always writing and nobody watches the tokens stream, every prompt piece is cut
  to 1,021 rows, and the operator may prefer longer gaps between a writer's tokens for faster prompt reading.

## Decision

1. Beside writers, when a whole chunk (`--chunk-rows`, 4,352) priced by the cost model costs at most 1.6 times
   `--prompt-budget-ms`, a prompt's piece is a whole chunk, or the rows left before the prompt's end or its next
   checkpoint if fewer. Otherwise pieces follow ADR-0044 as before, at most `whole_rows`.
2. While item 1 holds, another prompt may join a chunk past the fetch path, up to a whole chunk, while the plan's
   prompt time stays within 1.6 times the budget.
3. The cost model fits a second line, a fixed part and a part per row, from chunks of more than the fetch path's rows
   (`fetch_rows` less three), and prices such chunks on it once it has seen one. Until their sizes differ enough to
   split the line, its part per row is the fetch path's, or less if the line through 0 gives less.
4. After each prompt step, decode is still owed `decode_share` of its time (ADR-0017).

## Why

- A piece past the fetch path pays more fixed part and reads fewer rows a second until it nears a whole chunk
  (Context's table), so a piece that leaves the fetch path goes all the way.
- The budget is the operator's bound on a prompt step's time beside writers. Reusing it keeps one setting: the
  default fits no whole chunk, so its pieces stay as they were; a budget of 2,000 ms (3,200 ms at 1.6 times) fits the
  2,748 ms of a whole chunk at position 239,360, with room for the cost model's error.
- One line over both paths would price a 4,352-row chunk from 1,021-row pieces with the fetch path's smaller fixed
  part, and price those pieces from whole chunks' times seen with nobody writing.

## Options not chosen

- A new option for the piece size: the budget already says how long a prompt step beside writers may take.
- A piece of the rows that fit 1.6 times the budget on the long line: between 1,025 and about 1,565 rows at position
  0 it reads fewer rows a second than a 1,021-row piece and still holds the writers longer.
- A gate on the rows of the piece in hand instead of a whole chunk: at the default budget a prompt's last 1,022-1,385
  rows would become one chunk past the fetch path, a change to the default.

## Consequences

- At a budget that fits a whole chunk, a writer's longest gap is about a whole chunk's time plus up to 85 ms: about
  1.7 s at position 0 and 2.8 s near position 239k (*estimate*, from the records above).
- A whole chunk's price grows with the prompt's position, so a budget that fits one near position 0 may not fit one
  deep in a long prompt. Pieces then go back to `whole_rows`, and the long line learns a lower price only from the
  next long chunk, read with nobody writing. So the budget is set from the deepest position served.
- The cost model's file holds the second line, so a file kept by an earlier version is not read.
- `STATS` carries `long_chunk_ms` and `long_row_ms`.
- Not run on the cards yet: an ABBA run of a mixed load at the default and at 2,000 ms decides it.
