# ADR-0072: one thread at a time uses the cards' command lists

**Status:** draft. 2026-11-09.
**Revisit when:** a C ABI call needs the cards and can neither wait for the queued plans nor leave its work to the
engine thread, or a model needs a second thread on a list the engine thread uses.

## Context

- The core's engine thread runs the queued plans without holding the core's lock (`mu` in
  `core/api/engine_api.cpp`). The caller's thread makes the C ABI calls under that lock.
- The model's lists are shared by every slot: each card's immediate list `up`, which the page table's writes, the
  copies between pages, the swap rounds and the streamed KV's kernels use, and the commit list, whose fence
  `Model::wait_commit` waits on. `commit_pending_`, which says whether a commit is still running, is a plain member.
- Level Zero does not lock a command list, its queue or its fence. One thread at a time may use one.
- `engine_admit` called `Model::reset_slot`, and `engine_admit_images` called `Model::set_images`, on the caller's
  thread while the engine thread ran plans. The reset waits for the commit list and appends fills to `up`. The
  `engine_checkpoint`, `engine_restore` and `engine_sleep` calls already waited for the queued plans first (`drain`).
- A list's lost update, or a fence reset while another thread waits on it, can leave a list that the driver never
  reports finished, though its work is done. The engine then stops with "did not finish a list in 60 s". It is not
  shown that this caused the stops seen in service. The race is shown by the code.

## Decision

1. The model's calls, and with them every command list, fence and event, run on one thread at a time: the engine
   thread, or a C ABI call's while the engine thread waits idle.
2. A C ABI call on the caller's thread that needs the cards either waits for the queued plans to end (`drain`), or
   leaves the work to the engine thread, which does it at the start of its next plan.
3. `engine_admit` and `engine_admit_images` leave the slot's reset and its images to the engine thread
   (`Slot::reset_due`, `Slot::images_due`, `set_up`). `engine_checkpoint` and `engine_restore`, which wait, do a
   slot's pending work themselves before they read or write its state.
4. An error in the work left to the engine thread ends that plan with `ENGINE_EVENT_ERROR`, as any error in a plan
   does. `engine_admit_images` checks every image itself, so the model refuses none there.
5. `ze::List` checks the rule where `ENGINE_CHECK_LISTS` is set or `NDEBUG` is not: a call made while another thread
   is inside the same list throws.

## Options not chosen

- Every call that touches the cards waits for the queued plans, as checkpoint does. An admit would wait for the
  running plan and the queued ones, and the engine thread would then sit idle until the next plan came.
- A lock around each list. Each card's lists would need one, the waits inside them would hold it, and nothing would
  check that a new call takes it.

## Consequences

- An admit no longer waits for the cards. The reset moves into the next plan's time.
- A test runs the C ABI's code on a model of its own, without a card (`core/tests/engine_thread_test.cpp`), through
  `engine::open_engine` (`core/api/open_engine.h`).
- A new C ABI call that reaches the model states which way it follows item 2.
