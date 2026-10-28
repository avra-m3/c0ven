# ADR-0053: writing slots' next tokens ride in a prompt chunk as runs of one row

**Status:** draft, 2026-10-28; not accepted.
**Revisit when:** a writing slot's drafts can ride with its row, a prompt chunk stops staging every expert its rows
route to, or the writers' tokens beside an arriving prompt are measured to fall.

## Context

- A prompt chunk holds both cards for 0.5-1.4 s, and the writing slots get no token until it ends and a decode step
  follows. The chunk stages every uncached expert its rows route to, so a row added to it has no CPU misses, which
  are the part of a decode step that grows fastest with its rows.
- A chunk holds up to four runs (`PROMPT_MAX_RUNS`), and a prompt arriving beside writers usually fills one.
- Through the prompt path's kernels a writer's row cost more than the writer's own decode step once the writer was past
  12,288 positions: `p_kv16` converted its whole sequence to fp16 in each attention layer, and `p_attn_xmx` took its
  one row in work-groups sized for many rows
  ([measurements](../measurements/MEASUREMENTS.md#2026-10-28-writers-rows-in-a-prompt-chunk)).

## Decision

1. The core adds writing slots' next tokens to a prompt chunk while it has runs to spare, as runs of one row, lowest
   slot first. Each gets its token from the chunk's candidates, as from a decode step, and the draft layer reads the
   row. No drafts are made for them while a prompt is read, as for several writers in a decode step.
2. A run of up to 8 rows (`PROMPT_SHORT_RUN`) attends as the decode step does: from the 8-bit cache, over the cells
   the indexer chose, with no fp16 copy (`p_attn_rows`, `p_attn_rows_merge`). A prompt of 8 tokens or fewer read
   alone takes the same path.
3. The prompt kernels that follow a sequence run once for all of a chunk's runs. Only `p_kv16` and the long runs'
   attention, which share one fp16 copy of a sequence, run once a run.

## Why

With items 2 and 3, one to three writer rows add 5.6-9.8 ms (medians) to a chunk at any position, against 7.0-76.3 ms
through the prompt path's kernels and 20.4-58.4 ms for the writers' own decode step. Long runs keep their bits. Beside
arriving prompts the writers get one to two more tokens a prompt between them, with the prompt's first token as soon
as before. A prompt read alone takes 0.2% longer
([measurements](../measurements/MEASUREMENTS.md#2026-10-28-writers-rows-in-a-prompt-chunk)).

## Options not chosen

- Decode steps on a second queue beside the chunk: with the chunk on a low-priority queue and the steps on a
  normal-priority one, the steps starved the chunk until the driver reset the card.
- The writers' rows through the prompt path's own attention: it costs more than their decode step past 12,288
  positions (Context).

## Consequences

- A short run's tokens are the decode step's attention over the prompt path's products: not bit-identical to either
  path alone, judged by `greedy_test`.
- A writer in a chunk loses its drafts for the next step.
- Writers' drafts riding as further rows of their run are the next step; `p_attn_rows` takes up to 8 rows a run.
