# ADR-0074: swap rounds run beside a prompt while a slot writes

**Status:** draft. 2026-11-08.
**Revisit when:** an ABBA of writers beside long prompts finds the writers or the prompt slower than with rounds held,
or a lone prompt read with the cache following its routing is measured faster.

## Context

- The engine thread runs a swap round between plans: after a plan that ends a prompt, and otherwise on the cadence of
  `SwapCadence` (`core/api/swap_cadence.h`). Without this ADR, no round runs while any slot has prompt left to read.
- That hold was measured with one request and nobody writing, when a round held the engine thread about 27 ms:
  long-32k's prompt read 2,088.8 → 2,132.2 tokens/s with rounds held, long-4k's unchanged
  ([the record](../measurements/MEASUREMENTS.md#2026-10-16-swaps-and-prompts)).
- A round now holds the engine thread about 2.9 ms; the outgoing experts' copies and the next choice run beside the
  steps ([the record](../measurements/MEASUREMENTS.md#2026-10-20-swap-round-behind-the-steps)).
- Swaps leave every staging slot's arena piece in place, so the prompt path's copy lists stay valid
  ([the record](../measurements/MEASUREMENTS.md#2026-10-17-swaps-keep-the-copies)). A prompt chunk joins the thread that copies the
  outgoing experts before it reads the arena.
- With agent conversations, one slot is often reading a long prompt while others write. Under the hold the cache
  then stops following the writers for as long as some prompt is read, and the CPU computes more experts for each
  written token.
- Each prompt chunk adds its routing to the swap counts at 0.05 of a written token's weight (`kPromptRouting`). A
  140,000-token prompt read in about 100 s adds about 70,000 to each layer's counts, against about 40,000 from a writer
  at 40 tokens/s over the same time (*estimate*: tokens × 10 experts a token × weight). Beside writers the cache
  would follow the prompt.

## Decision

1. While a slot reads its prompt and no slot writes, no swap round runs, as before.
2. While any slot writes, rounds follow the usual cadence whether or not a prompt is read, and after a plan that ends
   a prompt. A round's swaps are shared among the plan's writers as before: 64 divided by their number.
3. A prompt chunk's routing joins the counts only when no slot was writing as its plan began. Beside writers it is
   dropped. A lone prompt's routing is counted as before, and the round at its end follows it.
4. Rounds beside a prompt take no further limit: the same swap count, and the same refills and evictions when the KV
   pool and the cache share the cards.
5. The rule is `swap_round_runs` and `count_prompt_routing` in `core/api/swap_cadence.h`, tested by
   `swap_cadence_test` without a card.

## Why

The hold saved a lone prompt about 2% when a round cost 27 ms of the engine thread. Beside writers, the misses the
held rounds leave cost every written token, and a round costs about a tenth of what it did.

The prompt's routing is dropped beside writers because the cache serves decode: a chunk stages every expert its rows
route to, so the cache does not speed the prompt itself. The counts a prompt leaves serve only its own first written
tokens, which count at full weight once it writes.

Each cost of a round beside a reader is small against a prompt chunk, which aims at 500 ms while a slot writes
(`prompt_budget_ms`):

- The engine thread: about 3 ms a round.
- The next prompt chunk waits for the outgoing experts' copies: up to 64 experts of about 2.05 MB, about 130 MB, or
  about 6 ms at the 22.4 GB/s one thread's `memcpy` moved in
  [the staging record](../measurements/MEASUREMENTS.md#2026-11-05-staging-through-a-mapping) (*estimate*).
- A round that changes the staged set records the prompt path's next copy list on the swap thread, about 45 ms there
  and 8 ms of wait ([the record](../measurements/MEASUREMENTS.md#2026-10-16-swaps-and-prompts)), and swaps alone never change it.
  Refills come only when more than four margins of pages are backed above those in use
  ([ADR-0045](0045-the-kv-pool-and-the-expert-cache-share-each-cards-memory.md) item 7), which a growing prompt rarely
  leaves. The evictions its KV needs happen with or without rounds, and a round plans them ahead of the need.
- The outgoing experts are read from the model file on the swap thread, about 130 MB a round, on the SSD the n-gram
  rows come from. That is 25-45 ms of the SSD every 4 to 16 plans at the 2.9-5.2 GB/s
  [read back from the file](../measurements/MEASUREMENTS.md#2026-10-25-arena-resident-and-tail-drop) (*estimate*).

## Options not chosen

- A lower prompt weight while writers exist. Any weight still lets a long enough prompt outweigh the writers, and the
  prompt gains nothing from the cache.
- Keeping a prompt's routing apart and adding it when that prompt ends. A chunk's counts mix every run in it,
  including writers' rows, so they cannot be split by slot without a count buffer per run.
- Fewer swaps a round, or no refills, beside a reader. None of the costs above is large against a chunk; the ABBA on
  the cards decides whether one is needed.

## Consequences

- `engine::Model::note_prompt_routing` takes whether to add the chunk's counts.
- Writers' rows that ride in a prompt chunk ([ADR-0053](0053-writers-rows-ride-in-a-prompt-chunk.md)) are counted
  with the chunk, so beside writers their routing is dropped too. Short prompt pieces read as decode windows still
  count at 0.05 through the decode step; they are at most a window long.
- Before this leaves draft, an ABBA on the cards checks a mixed load of writers beside long prompts (writers'
  tokens/s, CPU misses per decode token, prompt rows/s) and a lone long prompt (long-32k), which must not change.
