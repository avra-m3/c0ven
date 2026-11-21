# ADR-0080: exact choice-token scoring on temporary slots from cached prefixes

**Status:** draft, 2026-11-20; not accepted.
**Revisit when:** requests need full per-token logprobs across generated text, or choice scoring needs more than 16
candidate token ids.

## Context

- At the end of each step, each card runs `topk_rows` over its half of the vocabulary and returns only its 64
  highest-scoring tokens to the host (`core/models/qwen4exp/model.cpp`).
- When a client checks a proposed tool call or claim with a short multiple-choice prompt, an unlikely answer choice
  (below 0.1%) can rank outside the top 64. The host never receives its logit and cannot compute calibrated
  probabilities across the answer set.
- Running a check on a normal generation slot would either generate unnecessary tokens, trigger drafter passes, or save
  checkpoints that displace real session checkpoints.
- `c0ven` already keeps prefix checkpoints across turns ([ADR-0017](0017-scheduling-several-requests.md)) and batches
  up to 16 decode rows in one step ([ADR-0078](0078-a-decode-step-holds-16-rows.md)).

## Decision

1. Extend `engine_settings` with `flags`, `n_score_tokens`, and `score_tokens[16]` (`ENGINE_MAX_SCORE_TOKENS`), and add
   `ENGINE_SETTINGS_SCORE_ONLY`, `engine_slot_scores`, and `engine_read_slot_scores` (ABI version 24).
2. Read exact choice logits on demand via a single-group `gather_logits` kernel in `kernels/common.cl` launched on each
   card's immediate list after the prompt's final pass, before any draft pass reuses `card->logits`. Normal recorded
   step and prompt command lists stay unchanged.
3. Also gather each card's slice maximum and exp-sum in `gather_logits` and combine them in double precision on the
   host so callers receive both calibrated probabilities across the chosen tokens and exact full-vocabulary
   log-sum-exp.
4. For `ENGINE_SETTINGS_SCORE_ONLY` slots, skip sampling, drafting, lookup indexing, and post-prompt swap triggers.
   Batch concurrent short score-only prompt pieces into multi-slot decode windows of up to 16 rows (`STEP_MAX_ROWS`).
5. In the scheduler, `DECIDE` requests restore the longest matching checkpoint, reserve pages for the prompt only, take
   no new checkpoints, emit `SCORES` before `DONE`, and release their slot at once.
6. Expose `POST /v1/decide` in the front end with optional calibration temperature and `swap_order` position debiasing.
   When `close_thinking` is true (default), close an open `<think>` tag at the rendered prompt's tail so the leading
   conversation prefix still matches checkpoints taken during thinking turns.

## Options not chosen

- Raising `topk_rows` past 64 on every step: an unlikely choice can still rank outside any small top-k, and a wider
  sort adds work to every decode step.
- Recording `gather_logits` into the normal decode and prompt lists: would add a kernel launch and branch to requests
  that never ask for choice scores.
- Rewriting `enable_thinking` to `false` in `/v1/decide`: templates that place a reasoning-effort instruction at the
  start of the system message would diverge at token 0 and miss every saved checkpoint.

## Consequences

- Normal chat and completion requests execute identical command lists and take zero extra GPU work.
- A check appended to a checkpointed conversation reads only the check suffix, leaves existing checkpoints untouched,
  and frees its slot and KV pages immediately.
- Parallel swapped-order checks submitted together share the restored checkpoint's KV pages and batch in the same
  steps.
