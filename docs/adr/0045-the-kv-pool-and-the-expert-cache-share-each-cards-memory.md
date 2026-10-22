# ADR-0045: the KV pool and the expert cache share each card's memory

**Status:** draft, 2026-10-20; not accepted.
**Revisit when:** the cards' memory is managed by another layer than Level Zero's virtual memory, the pool's pages stop
being taken lowest first, or checkpoints should give their memory to the cache before the pool needs it.

## Context

- The KV pool is allocated whole at load, for the most tokens the slots may hold (`--kv-tokens`, or the slots times
  the context). The memory it takes would otherwise cache experts: with `--slots 4 --context 262144 --kv-tokens 512k`
  the pool takes 4.25 GB a card and the cache holds 248 experts a layer; with one slot, more. Most of the time the
  requests hold a small part of the pool.
- Each expert the cards do not cache is computed by the CPU when a token routes to it, and staged for every prompt
  chunk. More experts cached makes decode and prompts faster
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-16-more-experts-cached)).
- The pool's buffers and the cache's are read through addresses that recorded lists, the page table and the stage maps
  hold. Freeing device memory and allocating it again moves nothing only by chance, and lists recorded before such a
  free crash the driver past 16 GiB a card ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-16-vram-give-back)).
- Level Zero's virtual memory keeps a range of addresses in place while the memory behind it changes
  ([LEVEL_ZERO.md](../measurements/LEVEL_ZERO.md), row 32). On the B60 a 2 MB block is the page, a block moves between two ranges
  in 60-116 µs, recorded lists read the new memory, and copies through blocks run at the speed of plain device memory
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-20-kv-pool-and-expert-cache-share)).

## Decision

1. With swaps on and the expert count not set, the pool and the cache share each card's memory
   (`ENGINE_CONFIG_SHARE_KV`; the front end sets it unless `--deterministic`). Otherwise the pool is allocated whole,
   as before.
2. Each KV buffer and each layer's cache is a reserved range of addresses that never moves. A fixed part at its start
   is one allocation; past it, the memory is 2 MB blocks, each its own allocation, that move between the ranges and a
   card's list of spare blocks (`core/models/qwen4exp/kv_share.cpp`).
3. The cards always hold room for the whole pool. The caches' fixed parts hold the most experts a layer that leave
   room for the whole pool and the staging of every expert they leave out; the pool's fixed parts hold one prompt
   chunk. At load, every other block goes to the cache, each layer the same count.
4. Python takes the lowest free page first, and each plan says how many pages at the top of the pool hold nothing
   (`kv_free`). Before a plan runs, the core backs every page below them. If they are not backed, it takes blocks from
   the caches at once, and the plan waits.
5. Ahead of that, the core keeps a margin of backed pages above those in use: two prompt chunks' pages, and at least
   2,048. When less is left, it plans blocks for two margins, fills the leaving experts' places in the arena in the
   background, and moves the blocks between plans.
6. Each block comes from the layer whose experts it would put out are the least used by the swap rounds' counts; on a
   tie, from the layer that caches the most. The experts that stay move down into the slots the leaving ones free.
7. When more than four margins are backed above the pages in use, the blocks above two margins go back to the spare
   list. The swap rounds then put the most used uncached experts back, up to 64 a round, into empty slots and the
   spare blocks.
8. Every expert the cache leaves out keeps a place in the arena and a staging position; places exist for every
   expert the fixed parts leave out, and positions stay contiguous. A round that changes the staged set records the
   prompt path's next copy list behind the steps, as it fills the leaving experts' places; a change a plan cannot
   wait for records it before the next prompt chunk.

## Why

With 4 slots, a 262,144-token context and a 524,288-token pool, in six runs against six of main's in alternating
order ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-20-kv-pool-and-expert-cache-share)): the cache held 331 experts a layer at load and 321 with four
17,580-token prompts in, against 248. One request wrote 16.0% (gpqa-1) and 11.6% (gpqa-4) faster, four together
25.9%, a short prompt reached its first token in 0.484 s against 0.681, and 3.4k-7.6k-token prompts read 11.2%
faster. Each change of the split blocked the engine 14.3-31.7 ms, about a swap round. Recording the prompt path's copy
list on the engine thread after each change instead cost 0.37-0.41 s at the next prompt chunk, which is why a round
records it behind the steps (item 8).

## Options not chosen

- A pool sized by hand (`--kv-tokens` alone): a bet either way. Too small turns requests away; too large leaves
  experts uncached while it holds nothing.
- One allocation a card, the pool from the top down and the cache from the bottom up: moving the boundary needs the
  memory under one range of addresses to change, which only virtual memory gives without copying a region.
- Freeing and allocating device memory as the pool grows: the addresses would move under recorded lists and the
  page table, and the driver crashes past 16 GiB a card.
- Dropping checkpoints to give their pages to the cache: a checkpoint saves its prompt's reading; not measured.

## Consequences

- The number of experts cached changes while the engine runs. `/metrics` gives the number at load; the engine's log
  gives each change, its time, and the pages backed and in use.
- A checkpoint holds its pages until the pool needs them or a newer one takes its place (8 by default), so after long
  prompts the cache stays smaller until those checkpoints go.
- At its smallest the cache holds about what a whole pool leaves it.
- Giving blocks back to the spare list blocks the engine, 34.8 ms for 120 blocks a card
  ([MEASUREMENTS.md](../measurements/MEASUREMENTS.md#2026-10-20-kv-pool-and-expert-cache-share)); the refills that follow are swap
  rounds.
- `--deterministic` turns sharing off with swaps, so its tokens do not change.
