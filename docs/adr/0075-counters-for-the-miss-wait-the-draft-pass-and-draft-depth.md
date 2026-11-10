# ADR-0075: counters for the miss wait, the draft pass, draft depth and checkpoint evictions

**Status:** draft, 2026-11-10; not accepted.
**Revisit when:** an ABBA run finds the miss wait's clock reads cost a measurable share of a round, or a question
about decode time needs a counter these do not give.

## Context

- While serving, the engine thread is busy 0.997 of the time and the cards draw 117-126 W against a 220 W cap
  ([the record](../measurements/MEASUREMENTS.md#2026-11-10-serving-feed)). The counters of
  [ADR-0049](0049-counters-in-the-core-read-by-a-snapshot-call.md) say where most of that time goes, by plan part.
  Several of the largest parts inside a part are not counted.
- The wait for the CPU's misses is known only from a fit of a round's time over snapshots, near 10.6 ms of a 36.7 ms
  round (*estimate*, same record). The core reads the clock in that loop already, for the spin's time-out, but keeps
  no sum.
- The draft layer's pass after a step or round has a timing record (`ENGINE_TIMING_DRAFT`) and no counter. It shows
  only as part of the 0.084 of busy time that no counter covers.
- `layer_drafts` and `layer_accepted` give acceptance over all positions, 0.84, but not how it falls with a draft's
  position, and nothing says why a slot drafted as few as it did: 1.35 drafts a slot-run with three writers.
- The KV pool reached all its pages, and then two long prompts were read again with few or no cached tokens. Nothing
  says whether their checkpoints were evicted, or why.

## Decision

1. `engine_stats` gains, ABI version 23:
   - `step_miss_ns` and `round_miss_ns`: the engine thread's time, in plain steps and in rounds, from the moment it has
     seen every row's record of a layer with a miss to the moment it raises that layer's answer flag. The count is
     the existing `step_layers_missed` and `round_layers_missed`.
   - `draft_passes` and `draft_ns`: the `ENGINE_TIMING_DRAFT` records, summed where each is written.
   - `layer_drafts_1` to `layer_drafts_7` and `layer_accepted_1` to `layer_accepted_7`: of the draft layer's drafts,
     those verified and those accepted at each position of the slot's run.
   - `layer_stop_max`, `layer_stop_share` and `layer_stop_p`: of the slot-runs with the draft layer's drafts, those
     whose drafting reached `draft_max`, those that reached the fewer drafts the round's share gave the slot or whose
     window held fewer rows than its drafts, and those that stopped at a draft below `draft_min_p`. The reason is
     kept with the slot's drafts when they are made, and counted when the round verifies them.
2. The scheduler's `STATS` pairs gain `checkpoints_evicted_pages` (evicted because the KV pool could not give an
   admission's or a slot's growth's pages), `checkpoints_evicted_room` (evicted because the checkpoint pool was full),
   `cold_admits` (admitted with no checkpoint to restore, the prompt longer than `--chunk-rows`), and `kv_pages_used`,
   `kv_pages_slots` and `kv_pages_checkpoints` (the KV pool's pages in use, held by an admitted slot, held only by
   checkpoints). Each eviction writes an `INFO` line on the `engine.seqstate` logger with the checkpoint's id, its
   length in tokens and the cause.
3. `/metrics` derives the miss wait per step, per round and per layer with a miss, its share of the steps' and the
   rounds' time, the draft pass's mean and its share of busy time, `layer_accepted_at_1` to `_7`, and the shares of
   the draft layer's runs by where drafting stopped.

## Why

- The miss wait needs two clock reads for each layer with a miss, about 33 a round, where every row of every layer
  already reads the clock once before its spin. At tens of nanoseconds a read against a 36.7 ms round, that is
  far below a run's spread (*estimate*, from the clock's usual cost on x86-64; not timed on the cards).
- Every other counter is one addition where a timestamp or a count already exists, as ADR-0049 requires.
- The page split counts the slots' pages in a set when `STATS` is asked for, so the scheduler's plans pay nothing.

## Options not chosen

- A separate count of the layers with a miss for the miss wait: it would equal `step_layers_missed` and
  `round_layers_missed`.
- The miss wait taken from the next layer's spin timestamp, to save one read: the last layer has none, and the
  saving is not measurable.
- A histogram of acceptance by draft position kept in Python from the `DONE` lines: they carry per-request sums only.

## Consequences

- `engine/api/abi.py` mirrors the new fields; `test_abi.py` checks the struct's size of 63 counters. The mock leaves
  the new core counters zero, since it has no misses and no draft layer.
- A reader of two snapshots can say how much of a round the cards wait for the CPU, what the draft pass costs, how
  acceptance falls with depth and what limits drafting, and whether a long prompt read again followed an eviction.
