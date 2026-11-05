# ADR-0067: a prompt's last rows join its chunk, and the checkpoint takes its state partway

**Status:** draft, 2026-11-06; not accepted.
**Revisit when:** the checkpoint before a turn's reply moves, a decode window of a few rows costs much less than 80 ms,
or the cards write host memory much faster or slower.

## Context

- `qwen4exp`'s checkpoint rule marks a checkpoint just before the reply's opening (`engine/models/qwen4exp/turns.py`),
  so a later turn of the conversation restores it. A prompt's piece stops there, and the 5 to 7 rows after it are read
  as decode windows.
- Those windows cost 68-96 ms on one card, for any prompt length: a large part of a short prompt's first token
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-last-rows-in-the-chunk)).
- A checkpoint holds a slot's fixed state, 118,038,528 B: each DeltaNet layer's last 3 convolution rows and its
  recurrent state, and the n-gram embedding's last 9 rows. The kernels that write that state at a chunk's end can write
  it at any row as well.

## Decision

1. A prompt span may carry a mark: its rows read whole as one chunk, and the slot's next checkpoint takes the state as
   it stands after that many of them.
2. The scheduler marks a span only when the prompt is read alone and nobody writes, the span would have stopped at the
   prompt's last checkpoint, the rest of the prompt fits in the same chunk, the prompt has more rows than a window
   would take, and at least 9 rows come before the mark, so the state's history is the chunk's own.
3. The chunk's kernels write the state at the mark to host memory as they pass it: one checkpoint's size per card,
   allocated at load when the engine keeps checkpoints, and kept across sleeps. The recurrence writes it in whole
   lines, in a kernel of its own that only a marked chunk's list runs; other chunks run the recurrence as before.
4. `engine_checkpoint` with a mark waits for that chunk only, not for the plans queued behind it, and copies the state
   with the card's copy engine.

## Why

- First token on one card, against the windows: long-gen 0.80 → 0.75 s, gpqa-1 1.25 → 1.21 s, edit-py 1.60 → 1.53 s,
  doc-4k-qa 2.69 → 2.64 s; doc-32k, read after a short prompt only, 21.32 → 21.25 s. Decode per window is level or
  better on every request but one, 0.4% slower there
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-last-rows-in-the-chunk)).
- Host memory, not the card's: the cards' memory holds experts, and 118 MB is about one expert a layer.
- Whole lines: a lane of the recurrence holds one column, so its stores down the column, 4 bytes each, cost a marked
  chunk 200-1,100 ms in host memory. Turned through local memory, the stores are whole lines and cost a few ms.
- From inside the recurrence's loop, once: the same loop split in two stretches at the mark made every chunk about 2%
  slower, marked or not.
- A kernel of its own: the check for the mark at each row made the recurrence 3% slower in every chunk, 215 to 222 ms
  in a 4,352-row chunk on one card.
- The copy engine: a CPU copy out of this memory took 22-44 ms, against 4.5 ms for a checkpoint copied from a card.

## Options not chosen

- Keep the state at the mark in the card's memory and copy it after the chunk: 118 MB a card taken from the expert
  cache, about one expert a layer, which costs decode.
- Read the last rows in a second chunk: a chunk's fixed cost is larger than the windows'.
- Move the checkpoint after the reply's opening: a later turn would not find it, as its prompt differs there.

## Consequences

- A prompt's first token and its last rows come from the prompt path, not the decode path, so the first token's
  candidates differ in rounding from those of the same prompt restored at the checkpoint.
- The engine pins 118 MB more host memory a card when it keeps checkpoints, and records one more prompt list for
  each chunk size at load, for a chunk with a mark.
- Prompts read with others, or beside a writer, still read their last rows as windows.
