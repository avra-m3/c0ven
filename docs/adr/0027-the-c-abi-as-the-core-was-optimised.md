# ADR-0027: the C ABI as the core was optimised, against the design

**Status:** open, 2026-10-15: every item below needs a decision; none is made here.
**Revisit when:** each item is decided, by a new ADR that names the item it settles; or the core's ABI changes again.

## Context

[DESIGN.md, section 3](../DESIGN.md#3-the-c-abi-between-python-and-c) sets the interface between the control
plane and the core, and [ADR-0022](0022-two-processes-python-and-cpp.md) puts a C ABI there. The Python side,
`engine/api/interface.py`, follows DESIGN.md. `include/engine_api.h` version 2 departs from that interface in many
places.

`engine/api/native.py` has a thin binding of the version 2 calls (`CtypesCore`) and an adapter (`NativeEngine`) that
gives the scheduler the interface it is written against. Each difference the adapter bridges is a stopgap, not a
decision about the design. The items below list them, with what the adapter does.

The core's behaviour is cited from `core/api/engine_api.cpp`.

## Decision

None yet. Each item needs one.

1. **What a plan is.** DESIGN.md: a plan carries only changes, and the core keeps decoding its slots between plans.
   Version 2: a plan is one step, its prompt pieces and then one decode step for the slots it names; with no plan
   queued, the core does nothing. The adapter keeps up to two plans queued while any slot decodes, each naming every
   slot in decode. Open: keep the core stepless and let Python drive every step (DESIGN.md's "about one submit and
   one poll per step" still holds), or make the core decode on its own between plans.
2. **Who owns the KV cache.** DESIGN.md, section 4: Python keeps the page table, shares prefix pages by hash and
   sends page changes in each plan; the core writes them to both cards. Version 2: each slot has its own KV for the
   whole context, and no page crosses. The adapter drops the page changes. The scheduler's page bookkeeping still
   runs and still limits admission, from the page count of Python's memory plan, which the core does not use. Open:
   paged KV with prefix sharing, or fixed KV per slot; and, if fixed, what admission should count.
3. **Who plans memory.** DESIGN.md: `open` is given the tensor table, each tensor's split over the cards, and the
   memory plan (resident KV cells, experts cached, arena bytes, working bytes per card). Version 2: `open` takes the
   model's path and reads the GGUF itself; it decides what fits from `experts_per_layer` and `reserve_bytes`. The
   adapter passes the path, slots, context, checkpoints and CPU workers, and nothing of the memory plan. So the
   pinned host memory the core takes is not bounded by Python's plan, which matters on a host already short of
   memory. Open: which side plans memory, and which numbers cross.
4. **Sampling settings.** The protocol and `SlotSettings` carry min-p, the repeat, frequency and presence
   penalties, and their window. Version 2 samples with temperature, top-k, top-p and a seed only. The adapter drops
   the rest silently, so a request that asks for them gets output without them. Open: add them to the core,
   or refuse them in the front end.
5. **Ending a request.** DESIGN.md: the plan lists slots to stop, and each answers FINISHED, CANCELLED; end-of-text
   ids come with `open`. Version 2: the core stops a request itself at one of up to 8 stop ids in its settings, at
   `max_tokens` or at the context's end; nothing cancels a slot short of `release`. The adapter puts the model's
   end-of-text ids into each request's settings, and finishes a cancelled slot itself: it leaves decode, and its
   later events are dropped, but up to two queued plans still compute a row for it. The core's "context full" end
   has no protocol counterpart, and the adapter reports it as LENGTH. Open: a cancel in the ABI; where stop ids
   live; whether "context full" needs its own finish reason.
6. **Checkpoints.** DESIGN.md: `checkpoint`, `restore` and `drop` are ordered with the plans and do not wait; the
   core answers CHECKPOINT_DONE. Version 2: `checkpoint` and `restore` block until every queued plan has run, and
   answer by returning; `restore` needs a slot that has read nothing yet. Each checkpoint is sized for the whole
   context and pinned when the engine opens; the engine process asks for 8 by default. The adapter answers
   CHECKPOINT_DONE itself once the call returns, and remembers each checkpoint's length to know where a restored
   slot's prompt resumes. Open: blocking or ordered checkpoints; how many, and how large.
7. **Prompt pieces.** DESIGN.md: up to 8 spans a plan, each with its first position and a flag for the prompt's
   end. Version 2: up to 4 pieces a plan, each a slot and a token count, of at most `prompt_chunk` (4,352 by
   default); the core keeps each slot's position. With `prompt_chunk` 0 the core skips pieces and reads a prompt
   one token per decode step. The adapter splits spans into pieces, spreads them over as many plans as needed,
   checks each span starts where the slot's last one ended, and refuses a core whose prompt path is off. A slot
   joins decode only once the plan with its prompt's last piece is queued, since a slot named for decode before
   then would have its prompt read one token per step. Open: whether positions cross, and the limits.
8. **Slots.** `interface.py` allows 16 slots and the scheduler's options are built on that. Version 2 decodes at
   most 8 rows a step and says at most 8 slots decode together. The adapter refuses more than 8 slots. Open: the
   limit, and whether a slot past it may wait rather than be refused.
9. **Events.** DESIGN.md's event is 32 bytes: a plan serial, two values and the step's time on the cards. Version
   2's is 16 bytes, has no time, and has no STALLED or CHECKPOINT_DONE. Its ERROR names slot 0, never "no slot",
   so every error the core reports is one of the engine as a whole. The adapter rebuilds STEP_DONE's rows from the
   plan it queued and its time from the core's timing records, and passes every core error on as the engine's.
   Open: the event's fields, and whether an error can belong to one request.
10. **Timing.** DESIGN.md, section 7: a shared timing region, rings per thread in a memfd that `tools/attach.py`
    maps from outside the process. Version 2: `engine_read_timing` drains one record per prompt piece, decode step
    or draft round, with the experts the CPU computed. The adapter turns each into one STEP record and drops the
    miss count, and the attach tool cannot attach to the core. Open: the region, the drain, or both.
11. **Status codes.** `interface.ErrorCode` has 8 codes; version 2's `engine_status` has 5. The adapter maps
    BAD_ARGUMENT to INVALID_ARGUMENT, BUSY to WRONG_STATE (except a full plan queue, which it waits out) and the
    rest to INTERNAL. Open: one set of codes for both sides.
12. **Checking the ABI.** Version 2 reports its version only in `engine_info`, after the model has loaded, and has no
    call that reports it before `open`. The adapter compares `engine_abi_layout`'s table with
    `abi.py` when it loads the library, and the version after `open`. Open: a version call before `open`.
13. **Admitting a slot again.** `engine_admit` resets the slot's state on the caller's thread, and assumes no plan
    runs on it because the engine thread skips unused slots. A slot released while a plan runs on it is still being
    computed, and with drafting its catch-up runs outside the lock after the request ends. The adapter waits until
    every queued plan that named the slot has run before it admits into it again. Open: the core's guarantee, or the
    caller's rule.
14. **What the core can do that Python cannot ask for.** Version 2 has adaptive swaps (`ENGINE_CONFIG_SWAPS`),
    drafting with the MTP layer (`mtp_dir`, `draft_max`, `draft_min_p`), a routing profile, a card count, reserved
    device memory and a fixed experts-per-layer count. `OpenConfig` has none of them, and its `deterministic` switch
    ([ADR-0016](0016-how-a-change-is-judged.md)) has no counterpart. The adapter leaves the core's defaults: no
    swaps, no drafting. Open: which of these the engine process sets, and from what.

## Consequences

- The native backend loads the core and the control plane runs against it. `tests/host/test_abi.py` checks the
  layouts against the built core, and `tests/host/test_native.py` drives the adapter through the scheduler over a
  fake of the core that follows `engine_api.cpp`.
- DESIGN.md and `interface.py` describe the design, `engine_api.h` the core. Until these items are decided, the
  adapter is where the two meet; a change to either side changes it.
- An item decided in favour of the design changes the core and its ABI version; one decided in favour of the core
  changes DESIGN.md and `interface.py`, and shrinks the adapter.
