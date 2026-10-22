# ADR-0044: a prompt piece pays a chunk's fixed cost, so a short prompt is read whole

**Status:** draft, 2026-10-20; not accepted. Would supersede the first point of item 1 of
[ADR-0017](0017-scheduling-several-requests.md) where it says a prompt piece shrinks until its cost per row fits the
time budget.
**Revisit when:** a prompt chunk's fixed part stops being most of a short chunk's time, for example when the layers'
uncached experts are no longer staged for each chunk.

## Context

- A prompt chunk's time is mostly a fixed part: each chunk stages the layers' uncached experts, whatever its
  length. A chunk of 64 to 1,024 tokens took 650-765 ms
  ([measurements](../measurements/MEASUREMENTS.md#2026-10-17-prefill-timings)).
- A cost model that prices a prompt row by a moving average of the time per row cuts a prompt beside writing
  requests into pieces that fit the time budget (`--prompt-budget-ms`, 500 ms). A short prompt is then read as
  several pieces, the last ones as decode windows. Each piece pays the fixed part again and is followed by the decode
  it owes.
- With that model, a short prompt beside three writers takes 3.1-9.7 s to its first token
  ([measurements](../measurements/MEASUREMENTS.md#2026-10-20-short-prompt-whole)).

## Decision

1. The cost model fits a prompt chunk's time as a fixed part plus a part per row, by least squares over the chunks
   it has seen, older ones counting less. The engine reports each plan's chunks with its time, so the fit holds
   when the engine splits a plan.
2. Beside writing requests, a prompt of up to 1,024 rows is read whole when its rows cost no more than the fixed
   part, or when the whole prompt takes at most 1.6 times the budget.
3. A longer prompt is cut into pieces of the rows that fit the budget, but never fewer than the rows whose part
   equals the fixed part or 512 rows, whichever is fewer, and never more than 1,024 rows. A remainder under 64 rows
   joins the piece before it.
4. Beside writing requests, a prompt step waits until the step before it has reported its time, so the decode that
   step owes is counted before the next piece: the core can report a span done before its step.
5. The decode owed after a piece, and the rest of ADR-0017, stay as they are.

## Why

Beside three writers, the first token of a 228-568-token prompt came in 1.08-1.18 s, against 3.05-9.69 s, and the
writers lost about a third of the streamed output they lost with the moving average. With nobody writing, and for four prompts sent
together, the first tokens did not change ([measurements](../measurements/MEASUREMENTS.md#2026-10-20-short-prompt-whole)).

## Options not chosen

| Option | Why not |
| --- | --- |
| A larger time budget | at 1,200 ms with a quarter owed, the first token beside three writers still took 1.5-2.6 s ([measurements](../measurements/MEASUREMENTS.md#2026-10-20-short-prompt-whole)): a 568-token prompt was still cut, and every piece holds the writers longer |
| Reading any prompt whole | a long prompt would hold the writers for seconds |
| Pieces of a long prompt as long as their fixed part is most of their time | with a fit made from short chunks, a long chunk's rows look cheap, and pieces of 1,200-2,048 rows held the writers for up to 1.7 s, against 1.0 s before ([measurements](../measurements/MEASUREMENTS.md#2026-10-20-short-prompt-whole)) |

## Consequences

- A short prompt's piece can take up to twice the fixed part's time, longer than the budget; the writers wait that
  long once, not once for each piece.
- A long prompt's pieces are at most 1,024 rows, so its first token beside writers comes later than with whole
  chunks, and the writers wait no longer for one of its pieces.
- `engine/timing.py` holds the fit and the rule; the core's timing records give a plan's chunks.
