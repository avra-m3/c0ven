# ADR-0065: past the trained context the rotation is scaled by YaRN

**Status:** draft, 2026-11-02; not accepted. Item 3 of [ADR-0006](0006-scope.md) superseded in part.
**Revisit when:** the model's answers past 262,144 tokens are measured and found poor, or a request needs its own
scale.

## Context

- [ADR-0006](0006-scope.md) item 3 sets the context at 262,144 tokens, the length the model was trained on.
- Strata runs longer contexts with YaRN. It picks `yarn` by default above 262,144 tokens, with
  `s = context / 262,144`, and refuses no scaling there
  ([MODEL_LAYER.md, section 3.3](../spec/MODEL_LAYER.md#33-rope-821-832)). Its YaRN is llama.cpp's, and llama.cpp's CPU
  backend is the correctness reference ([ADR-0012](0012-correctness-reference.md)).
- K is stored in the cache after its rotation, and each pooled key is rotated when it is built. A cached sequence
  therefore holds for one `s`.

## Decision

1. A context above 262,144 tokens rotates by YaRN, with the parameters in
   [MODEL_LAYER.md, section 3.3](../spec/MODEL_LAYER.md#33-rope-821-832). At 262,144 tokens and below the rotation is
   unchanged, bit for bit.
2. The longest context the engine accepts is 4 times the file's `context_length`: 1,048,576 tokens in every published
   file. The factor is the engine's limit, and nothing past it has run.
3. The core takes `s` from `max_ctx` and the file's `context_length` at load, and the pairs that scale from
   `context_length`, `rope.dimension_count` and `rope.freq_base`. YaRN's `beta_fast`, `beta_slow` and the 0.1 in its
   magnitude are fixed by the method. There is no flag and no change to the C ABI.
4. `s` applies to every request on the server, whatever its length.
5. The main layers' Q and K, the indexer's queries and its pooled keys scale. The MTP layer and DFlash 2 do not: they
   attend over a short window of positions, far inside the trained distances.
6. Past 262,144 tokens the memory plan's estimate of working memory grows with the context.

## Why

This is what Strata and llama.cpp do, so the reference can check it. An `s` set at load keeps every cell in the cache
valid for every request.

## Options not chosen

- **An `s` for each request.** A short request would keep the trained rotation, but the prefix cache would split by
  `s`, and a conversation that outgrew its `s` would have to read its whole prompt again.
- **No scaling.** Positions up to 262,144 would not change, but past that the angles leave the trained range. Strata
  refuses it.
- **Linear scaling.** It also compresses the fastest pairs, which tell neighbouring positions apart.

## Consequences

- Reading a prompt slows with its position, and a card caches fewer experts as the context grows
  ([MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md)). Neither is measured past 262,144 tokens.
- A server above 262,144 tokens answers short requests differently from one at or below it.
- The model's answers past 262,144 tokens are not measured here.
