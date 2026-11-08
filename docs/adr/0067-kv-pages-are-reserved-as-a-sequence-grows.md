# ADR-0067: KV pages are reserved as a sequence grows, and a full pool preempts the newest slot

**Status:** draft. 2026-11-08.
**Revisit when:** preemptions are seen to read back more prompt tokens than the up-front reservation's waits cost, or
the core can keep a stopped slot's pages for its return.

## Context

- A request reserved the KV pages of its whole prompt plus `max_new` when it was admitted
  ([DESIGN.md, section 4](../DESIGN.md#4-sequence-state-the-kv-cache-and-the-rest)). The front end cuts `max_new`
  to the context's end (`budget` in `frontend/generate.py`), and agent clients send one large `max_tokens` on every
  turn, so every request reserved close to a whole context.
- With 3 slots of 262,144 tokens and a 589,824-cell pool, prompts of about 68K, 103K, 138K and 154K tokens with
  `max_tokens` 131,072 reserve 199K to 262K cells each. Two fit, so the third slot stays idle while requests wait.
  Each admission also evicts the checkpoints of the conversations that wait, so their next turn reads the whole
  conversation again. The prompts themselves total about 460K cells.
- The scheduler already maps a writing slot's pages ahead in blocks, and the core waits with `STALLED` when a slot
  reaches a position with no page ([ADR-0033](0033-paged-kv-kept-by-python.md), items 4 and 7).
- A sampled token depends only on the seed, its position and the tokens before it, the penalties counting the last
  `penalty_last_n` of those ([MODEL_IO.md, 2.4](../spec/MODEL_IO.md#24-the-sampler); `uniform` in
  `core/api/sampling.cpp`).

## Decision

1. Admission reserves the prompt's pages, less those shared with the checkpoint it resumes from, and one block of
   `decode_lookahead + decode_batch` cells after the prompt, at most `max_new`. The per-class counts of
   [ADR-0033](0033-paged-kv-kept-by-python.md) item 3 stay.
2. A writing slot reserves further pages a block at a time, when its mapped cells run short of its lookahead.
3. When the pool cannot give a block, checkpoints are evicted that hold the last reference to a page of the class
   that is short: least recently used first, kept ones last, never one a slot is restoring from. A checkpoint whose
   pages are all shared frees nothing and stays.
4. When the pool still cannot give a block, the slot keeps its mapped cells. Once it stalls, it takes the one page it
   waits for if the pool has it. If not, the scheduler stops the newest slot of the lowest priority, reading or
   writing, and the stalled slot itself if that is the newest. One slot stops at a time.
5. A stopped slot's request goes back to the queue with its arrival time, which puts it ahead of every request that
   came after it. The tokens it wrote are added to its prompt and taken from its `max_new`. They are not sent to the
   front end again, and no `RESUME` is sent for the new admission. Its `DONE` counts the client's prompt and every
   token written, with the first admission's times and reused tokens.
6. A request preempted while the engine drains for sleep is not refused with the requests that were never admitted.

## Why

The oldest slot is never stopped while another slot runs, and a slot running alone has the whole pool once the
checkpoints are evicted, since the pool holds at least one context (`--kv-tokens`). So the oldest slot always
finishes and the queue moves. vLLM preempts the same way, by recomputing the newest request.

Reading the written tokens as prompt gives the next token the same position, the same tokens before it and the same
seed, so the sampler's inputs are unchanged, and in the mock the tokens are identical. On the cards the written
tokens' rows are then computed by the prompt path, not by the decode steps that first computed them, and the two
are not known to round alike. A later token can then differ as it can when the work is split another way. This is
not checked on the cards.

## Options not chosen

- **Keep the up-front reservation and ask clients for a smaller `max_tokens`.** The clients send one value for every
  turn, and the server cannot know what a turn will write.
- **Wait until every writing slot has stalled before stopping one.** The stalled slot would wait for others to finish
  or stall, and a newer slot could take the pages it waits for first.
- **Stop only writing slots.** A reading slot holds its whole prompt's pages, and the oldest writer, alone with
  readers, would have to stop itself.
- **Keep the stopped slot's pages as a checkpoint for its return.** It needs a checkpoint at the stop's position, and
  the pages are what the pool is short of.

## Consequences

- `SeqState.grow` reserves pages past the admission and returns whether it could; `PagePool.spare` gives the pages
  each class can still reserve.
- A request whose prompt alone does not fit the pool waits or is refused as before.
- `plan_admission` still evicts every checkpoint it can when the request will not fit even then, and `_admit` stops
  at the first waiting request that does not fit while a smaller one could take a free slot. Both remain to be
  changed.
