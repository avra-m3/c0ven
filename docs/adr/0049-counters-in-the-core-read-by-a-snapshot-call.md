# ADR-0049: the core counts what it runs, and a snapshot call reads the counters

**Status:** draft, 2026-10-25; not accepted.
**Revisit when:** a counter costs a measurable share of a step, or a reader needs a counter per layer or per expert.

## Context

- `/metrics` reported the engine's facts from its `INFO` line and the front end's request counters. Nothing said how
  the engine spent its time: how many experts the CPU computed a step, how often the drafts were accepted, whether
  the draft layer or the slot's own tokens gave more tokens for the time, how much the swap rounds held the engine
  thread, or whether the cards waited for Python.
- The core already measured most of it. Its timing records carry each step's rows, time and misses
  ([DESIGN.md, section 7](../DESIGN.md#7-timing-and-debugging)); `lookup.cpp` keeps each slot's accepted drafts by
  source, decayed, to choose between the sources ([ADR-0041](0041-drafts-from-the-slots-own-tokens.md)); the swap
  module counts its swaps; the n-gram table counts its rows, hits and reads. None of it left the core except the
  timing records, which the native adapter drained and summed into `STEP_DONE` events, dropping the miss count
  ([ADR-0027](0027-the-c-abi-as-the-core-was-optimised.md), item 10).
- The host must stay light during decode: the CPU's expert workers share the RAM's bandwidth with the cards, and a
  competing load slows them 1.6-2.2 times ([the design-inputs record](../measurements/MEASUREMENTS.md#2026-10-13-design-inputs)).
  Per-kernel tracing (`Options::trace`) and the routing dump are tools, not metrics.

## Decision

1. The core keeps counters since open in `engine_stats`, one addition each where the timing record is already
   written under the engine's lock: plans and the engine thread's time in them; prompt chunks and windowed pieces
   with their rows and time; decode steps without drafts, with rows, time, misses and layers missed, and how many of
   them had drafting on and every row writing; draft rounds the same; each slot's run in a round by the source of its
   drafts, with the drafts verified and accepted and the rounds' time by source; swap rounds, swaps and their time;
   stalls, checkpoints and restores; the n-gram table's rows, hits and reads; the experts cached per layer at the
   last plan's end. Per slot, `engine_slot_stats` holds its steps, rounds, drafts and accepted drafts, and the part
   from its own tokens.
2. `engine_read_stats` copies the counters out, awake or asleep; `engine_read_slot_stats` copies an admitted slot's. ABI
   version 9. A reader takes two snapshots and differences them; the core keeps no windows or rates.
3. The engine process answers a `STATS` command with a `STATS` line: the core's counters, field for field, and the
   scheduler's (requests waiting, reading and writing; the cost model's decode, chunk and row parts; the decode time
   owed). Each `DONE` line carries its request's wait for a slot, its steps, its drafts and the accepted ones, and the
   part from its own tokens, as `key=value` pairs after `reused`.
4. `/metrics` reports the `STATS` line's values as numbers, the request counters in its totals and recent requests,
   and the ratios read from the counters in `frontend/metrics.py`. The core counts; Python divides.

## Why

- Three ABBA rounds of the branch against its base, six fresh servers an arm at 4 slots and a 262,144-token context,
  found no cost: prompt rates within 0.3%, decode medians 0.995-1.043 of the base's on seven of eight prompts (the
  eighth, long-gen, 0.905 in the series and 1.10 alone), four writers together 93.9 against 93.3 tokens/s
  ([the record](../measurements/MEASUREMENTS.md#2026-10-25-counters-abba)).
- The same record reads the counters: the cards hold 331 experts a layer at load and 328 at the end, with the KV
  pool's pages costing 3-4 of them (the floor the load keeps room for, 246, is reported beside the count); a decode
  row misses 66 experts; the engine thread is busy 99.3% of the time; rounds with drafts give 112-126 tokens a second
  of card time against 67 for plain steps. On one card 104 experts a layer are cached and a row misses 243. None of
  this was visible before, and the first reading of the series mistook the floor for the count, which is why the
  struct carries both.

## Options not chosen

- Draining the timing records in the engine process and summing them: a read on the engine's lock at every poll,
  and the records have no source for a round's drafts, no swap times and no counters that are not steps.
- Rates or moving windows kept in the core: a scraper differences snapshots itself, and the core's lock holds no
  arithmetic it need not.
- Counters per layer or per expert: a hit histogram per step is many additions, and the questions asked are answered
  by the sums.
- The counters on the `STEP` event: an event per plan is already sent; a snapshot is read only when asked.

## Consequences

- The ABI gains two structs and two calls; `engine/api/abi.py` mirrors them and `test_abi.py` checks the layouts.
- The mock fills the counters it has (plans, steps, rows, the times it reports) and leaves the rest zero.
- A tool that compares two `/metrics` snapshots can say, for a span of requests: misses per decode row, tokens per
  second of card time in plain steps against rounds with the draft layer's drafts and rounds with the slot's own
  tokens, drafts accepted per run by source, the swap rounds' share of the engine thread, and whether the cards
  waited for Python.
