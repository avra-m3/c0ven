# ADR-0039: the core samples with every setting a request carries, in llama.cpp's order

**Status:** draft, 2026-10-19; not accepted. Would settle item 4 of
[ADR-0027](0027-the-c-abi-as-the-core-was-optimised.md).
**Revisit when:** a request needs a sampler llama.cpp's default chain does not have, or the cards return more or fewer
than 64 candidates.

## Context

- The protocol and `SlotSettings` carry min-p, the repeat, frequency and presence penalties, and their window. ABI
  version 6 samples with temperature, top-k, top-p and a seed only, and the adapter drops the rest silently
  (ADR-0027, item 4). The OpenAI API takes `min_p`, `presence_penalty`, `frequency_penalty` and
  `repetition_penalty`, so a client that sends them gets output without them.
- ABI version 6 applies the temperature before top-p, as transformers and vLLM do. The correctness reference is llama.cpp
  at `3cf0325` ([ADR-0012](0012-correctness-reference.md)), whose default chain takes top-p and min-p on the logits
  before the temperature ([MODEL_IO.md, 2.4](../spec/MODEL_IO.md#24-the-sampler)). Strata samples in that order too
  ([STRATA_DECODE.md, section 6](../spec/STRATA_DECODE.md#6-sampling)).
- The core samples on the host from the 64 largest logits the cards return (DESIGN.md, section 9). Logits never
  leave the cards.

## Decision

1. `engine_settings` carries `min_p`, `penalty_last_n`, `penalty_repeat`, `penalty_freq` and `penalty_present`, ABI
   version 7. The native adapter passes them.
2. The core samples in llama.cpp's default order: the penalties, top-k, top-p, min-p, the temperature, each with
   llama.cpp's rule.
3. The penalties count the last `penalty_last_n` tokens before the one sampled, the prompt's included and image and
   video tokens not, as llama.cpp's server counts them. A window row and a draft count the rows and drafts before
   them, so a token is the same whether it was sampled in a window or alone.
4. The penalties change the logits of the cards' 64 candidates only.

## Options not chosen

- **Refusing min-p and the penalties in the front end.** Clients send them, and the core needs only the host's
  candidates and the tokens it already keeps to apply them.
- **The temperature before top-p.** It matches transformers, but not the reference the engine is checked against.
  With the model's defaults (temperature 1, no penalties) the two orders give the same tokens.
- **Penalties over the whole vocabulary on the cards.** It needs each slot's recent tokens on the cards and a
  kernel over 248,320 logits a row. A penalty only lowers a logit (for a repeat penalty of at least 1 and the other
  two at least 0), so a candidate it lowers can fall below a token the cards did not return. The k tokens top-k keeps
  are still the right ones while at most 64 − k of the candidates are penalised.

## Consequences

- ABI version 7: a core and a Python side of different versions refuse each other at open.
- Drafts are picked with the request's sampler, penalties included, so a penalty does not make the main model reject
  them by itself.
- ADR-0027 item 4 is settled.
