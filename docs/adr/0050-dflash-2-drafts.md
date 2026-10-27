# ADR-0050: the engine drafts with DFlash 2 when its files are present

**Status:** draft, 2026-10-27; not accepted. Supersedes item 3 of
[ADR-0029](0029-the-engine-process-drafts-and-swaps.md) in part (where the engine process looks for the drafter).
**Revisit when:** several writing slots decode slower with DFlash 2 than with the MTP layer, or a window of 5 to 8
rows costs about what a window of 4 does.

## Context

- The engine drafts with the model's MTP layer ([ADR-0029](0029-the-engine-process-drafts-and-swaps.md)). Its first
  draft comes from the round's catch-up, and each further draft costs one more pass, so a lone slot takes up to 3.
- [`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) is a separate drafter
  for this model: five dense BF16 layers that read the main model's hidden states after layers 5, 15, 25, 35 and 45,
  and draft a block of up to 7 tokens in one pass. Its files are 0.9 GB without its candidate selector, against the
  MTP layer's 5.2 GB ([MODEL_FILES.md](../MODEL_FILES.md#the-dflash-2-drafter)).
- On the engine's own text, scored offline, DFlash 2's first three drafts are accepted about as often as the MTP
  layer's: 0.889, 0.812 and 0.732 against 0.872, 0.802 and 0.748 greedy. Its 4th to 7th are accepted 0.670 to 0.525
  ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-acceptance)).
- A step's window holds up to 8 rows. A window of 8 rows costs 35.8 ms against 23.6 ms for 4
  ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-on-the-cards)).

## Decision

1. The core drafts with DFlash 2 when the draft directory holds `dflash-manifest.json`, and with the MTP layer
   otherwise. `engine_config.mtp_dir` names either; the ABI does not change.
2. The engine process looks for `dflash/` beside the model file, then `mtp/`, each with its manifest.
   `--draft-dir` and `--no-draft` work as before.
3. A lone slot drafts up to 7 tokens with DFlash 2, and stops at the first draft below `draft_min_p`. Several writing
   slots take up to 2 drafts each, as with the MTP layer.
4. Each decode step and prompt chunk keeps the five layers' hidden states for its rows. The drafter's context is a
   ring of each slot's last 2,112 positions. After a prompt it starts from the prompt's last 8 rows, as the MTP layer
   does.
5. A draft is picked from the main model's head over the drafter's row, as the request samples, with that position's
   uniform. The checkpoint's candidate selector is not used.
6. `tools/dflash_fetch.py` fetches the drafter at a pinned revision and checks its sha256.

## Why

On the cards, one slot at a 262,144-token context with 300 experts cached a layer in both arms, DFlash 2 decodes six
reference prompts at 107.2 tokens/s against 103.5, with the same tokens (6 loads an arm, ABBA). The gain is +36% on
edit-py, +3.0% on gpqa-1 to gpqa-4, and none on agent-turn1. A window gives 3.00 tokens against 2.52, and costs 27.9
ms against 24.3. The pass costs 4.0 ms a window, of which the main model's head takes 1.09 ms and the BF16 products
about 1.39 ms ([the record](../measurements/MEASUREMENTS.md#2026-10-26-dflash2-on-the-cards)).

Without a cap, the cards cache 356 experts a layer with DFlash 2 against 346 with the MTP layer, from the same
memory.

## Options not chosen

- Keeping the MTP layer: slower for a lone slot, and it leaves the cards fewer experts.
- The candidate selector. Scored offline, 7 drafts with the 0.5 bar give as much without it (+13.0% greedy against
  +12.3% with it), and it is one more pass.
- The context from the prompt's last 2,048 rows. Offline it adds 3-4 points of acceptance from the third draft on,
  but each prompt then costs a context pass over its rows.

## Consequences

- A deployment drafts with DFlash 2 once `dflash/` is beside the model files. Without it, the engine drafts with the
  MTP layer as before.
- Rounds of several writing slots with DFlash 2 are not measured. They take 2 drafts each, where its one pass
  saves less over the MTP layer's chain.
- A pass's head and BF16 products run near the cards' memory bandwidth. A faster pass would need fewer rows through
  the head, or a smaller drafter.
