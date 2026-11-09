# ADR-0072: on one card, the CPU computes a short chunk's rarest experts

**Status:** draft, 2026-11-06; not accepted. Depends on [ADR-0065](0065-a-short-chunk-writes-through-a-mapping.md).
**Revisit when:** a copy of an expert to the card costs much less or much more than 94 µs, the CPU's cost per row changes,
or two cards should do the same.

## Context

- After [ADR-0065](0065-a-short-chunk-writes-through-a-mapping.md) a short chunk still spends most of its time copying
  experts to the card, at 94 µs for a 2.05 MB expert, and that rate is what the card's link allows.
- Many of the experts it copies serve few rows. On one card with 137 experts cached a layer, a 212-row chunk reads 178
  uncached experts a layer: 46 picked by one row, 27 by two, 31 by three or four
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-rare-experts-on-the-cpu)).
- The CPU computes such rows for about 40 µs an expert and 17 µs a row, with its 16 workers, while the copies of the other
  experts go on.

## Decision

1. On one card, in a chunk of `kFetchRows` or fewer, an uncached expert that `kCpuRows` (3) rows or fewer picked is computed
   by the CPU and not staged.
2. `p_moe_scan` gives such an expert no tiles. `fetch_experts` does not copy it after the router; one copied ahead of the
   router is left unused.
3. The chunk's list copies each layer's `xg` to host memory after `p_moe_scatter`, and waits before the staged products
   until the CPU has finished the layer.
4. `rare_experts`, on the engine thread and the workers, computes each layer's in turn with `compute_many`, and writes their
   rows of `yg` through the CPU's mapping of the region, ending as ADR-0065's writes end.
5. Where the staging buffer or the region cannot be mapped, the CPU computes no expert of a chunk.

## Why

- First token against ADR-0065 alone: long-gen 0.65 → 0.48 s, gpqa-1 1.04 → 0.81 s
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-rare-experts-on-the-cpu)).
- 3 rows: 2 is best on the shortest prompt and 4 on a longer one, by a few hundredths of a second, and at 6 the CPU's time
  takes back what the copies save
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-11-05-rare-experts-on-the-cpu)).
- The CPU's rows round a little differently from the card's. `kl_test` at 500 prompt tokens on one card moves mean KL by
  -0.0001 (95%: -0.0070 to +0.0068) and top-1 agreement by -0.44 points. This is the change's cost.

## Options not chosen

- The same on two cards, not tried: two cards cache more than twice the experts a layer, so fewer are left to copy, and each
  copies half an expert.
- A limit that follows the chunk's rows, 2 for the shortest and 4 for the rest, not tried: a few hundredths of a second, for
  one more thing to tune.

## Consequences

- A short chunk's results differ in rounding from a chunk computed wholly on the card. Which drafts a request tries can
  change with it, so decode moves a little either way by request: in the measured set from -3% to +2%.
- Which experts the CPU computes depends on what shares the chunk, so a prompt's numbers read with others differ in
  rounding from the same prompt read alone. `--deterministic` turns this off, as it does swaps, and `pack_test`, which
  asks for an exact match, runs with it off.
- The workers run during a short chunk on one card.
- A chunk's list holds one more copy and one more wait a layer on one card, and the event pool two more kinds.
