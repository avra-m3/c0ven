# ADR-0054: the BF16 checkpoint is the end-to-end reference, by KL on a fixed set

**Status:** draft, 2026-10-29; not accepted. Would supersede item 2 of [ADR-0012](0012-correctness-reference.md) (the end
to end reference). Items 3 and 4, the layer and kernel references, stay.
**Revisit when:** Qwen publishes a newer checkpoint, vLLM's reading of this one is found to be wrong, or the engine
loads a file other than IQ3_S ([ADR-0038](0038-the-other-quants-load.md)).

## Context

- End to end, the engine was compared with llama.cpp's CPU backend on the same IQ3_S file, by `greedy_test`: is the
  engine's top token llama.cpp's greedy token, on 861 positions of prompts of up to 512 tokens. That counts top-1
  agreement only. It says nothing of how far the rest of the distribution moved, and it cannot see what the IQ3_S
  file itself loses.
- llama.cpp's own result moves with its batch size: a mean KL of about 0.035 and 91% top-1 between `-ub` settings
  ([REFERENCE_OUTPUTS.md](../measurements/REFERENCE_OUTPUTS.md#what-changes-it)), more than the 0.02 gate of ADR-0012.
  The engine's prompt path and its decode path differ from each other by the same amount
  ([record](../measurements/MEASUREMENTS.md#2026-10-29-kl-bf16)).
- Changes that trade precision for speed (skipping the smallest routed experts, int8 weights for the
  hyper-connections) need a measure of what they cost the model, against the model.
- Qwen publishes the BF16 checkpoint, `Qwen/Qwen3.8-Flash-Next`: 335 GiB, too large for the test machine. vLLM reads
  it on four H200s and returns each prompt position's log-probabilities.

## Decision

1. **The end-to-end reference is Qwen's BF16 checkpoint** at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`,
   read by vLLM 0.32.0 in BF16 with tensor parallelism 4, teacher-forced. Its output on the KL set is the base dump,
   made once and kept beside the other reference data ([REFERENCE_OUTPUTS.md](../measurements/REFERENCE_OUTPUTS.md#the-kl-set)).
2. **The KL set** is five segments of 1,000 tokens in `reference/kl-set.jsonl`: code, documentation, English prose,
   German, French, Japanese and Chinese, and a coding agent's turns.
3. **The comparison** is `kl_test`: the engine reads each segment one token a step, and at each of the 4,995
   positions keeps its log-probability of the next token and of its 1,024 most likely tokens. It reports top-1
   agreement, mean KL, p99 KL and perplexity against the base, overall and per segment.
4. **The gate** is mean KL at most 0.170 and top-1 agreement at least 83.5%. The engine on main gives 0.158 and
   84.7%. The gate catches breakage; it is set again when the engine's figure moves.
5. **A change that moves the distribution** is judged by `kl_test --ab`: the base, the engine before and after, and
   the change in mean KL with its 95% interval over the same positions. A change of rounding alone moves it by about
   0.002.

## Why

- The BF16 checkpoint is what the model is. Against it, the KL counts the file's loss and the engine's together, and
  a change's cost is the difference it makes to that figure. llama.cpp on the same file gives 0.160 against it, the
  engine 0.158: on main the figure is the file's loss
  ([record](../measurements/MEASUREMENTS.md#2026-10-29-kl-iq3s-llamacpp)).
- KL over the whole distribution sees changes that keep the top token, and per-segment figures show which kind of
  text a change hurts. The agent segment gives 0.359, against 0.087 to 0.148 for the others.
- vLLM gives the same bytes on two passes, and the engine gives the same bytes on two runs, so the figures have no
  run-to-run noise. Their noise is the set's sampling error: 0.0065 on the mean KL, 0.5 points on top-1.

## Options not chosen

- **llama.cpp on the IQ3_S file as the base.** Its result depends on its batch size by more than the gate, and a base
  on the same file cannot show what the file loses.
- **The top 64 tokens only.** On distributions whose top 64 hold 80 to 90% of the mass, KL from them reads 15 to 20%
  low; from the top 1,024, within 3%. Both sides hold every logit, so the larger K costs only the dump's size, 41 MB.
- **Positions read through the prompt path.** vLLM reads each segment as one prompt; the engine's decode path is the
  one every generated token takes. `kl_test` with a prompt length reads the first tokens through the prompt path for
  a check of that path; the KL to the base is the same within its error (0.1775 against 0.1755 on the same 2,500
  positions).

## Consequences

- `greedy_test` is removed. `kl_test` is the level-5 check of [TESTING.md](../TESTING.md), with the base dump in
  `FLASH_REF_DIR`; it skips without it.
- llama.cpp's greedy tokens stay in `reference/` for `api_test`, `route_profile` and the `reference` engine of `engine/api/reference.py`.
- The KL base of llama.cpp (8.1 GB, the 0.02 gate) is no longer read by any check.
- A change judged by `greedy_test` before, a change of rounding among them, is judged by `kl_test --ab`.
