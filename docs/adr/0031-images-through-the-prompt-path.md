# ADR-0031: images in scope, encoded through the prompt path's buffers

**Status:** draft, 2026-10-17; not accepted. Would supersede item 2 of [ADR-0006](0006-scope.md) in part ("Vision ... out of scope") and extend
[ADR-0012](0012-correctness-reference.md) to the image encoder.
**Revisit when:** the encoder's time at one megapixel measures well above the estimate below, the staging slots
measure too small on the deployed settings, or pinning the encoder's 0.90 GB fails at load.

## Context

- **The model sees images.** Qwen3.8-Flash-Next's checkpoint is `Qwen4ExpForConditionalGeneration`, with a 27-layer
  image encoder. ISTA ships it as `mmproj-Qwen3.8-Flash-Next-BF16.gguf`, 907,543,008 B, beside the text files
  ([the reference record](../measurements/MEASUREMENTS.md#2026-10-17-encoder-reference)). Its maths is
  [MODEL_VISION.md](../spec/MODEL_VISION.md).
- **llama.cpp runs it.** The pinned commit's `tools/mtmd` loads the file as `qwen3vl_merger` and has a CPU graph for
  it (`tools/mtmd/models/qwen3vl.cpp`).
- **Decode is bound by the experts on the cards.** An encoder resident in VRAM, split as the text model is, would take
  449 MB per card, 439 experts at the mean size, and cost about 0.3-0.6 ms on a 17.3 ms token (*estimate*, from
  [the hit-rate record](../measurements/MEASUREMENTS.md#2026-10-12-expert-hit-rates)).
- **The prompt path's buffers sit idle between chunks.** Each card has two staging slots for the uncached experts and
  activation regions sized for a chunk of 4,352 rows; none is in use between one chunk's lists and the next's
  ([DEVELOPING.md](../DEVELOPING.md#how-a-prompt-chunk-runs)).
- **The encoder is compute-bound.** 0.90 GB of weights against about 5.6 TFLOP for a one-megapixel image
  (*computed*: [MODEL_VISION.md](../spec/MODEL_VISION.md#8-what-it-costs)).

## Decision

1. **Images are in scope** for Qwen3.8-Flash-Next, from ISTA's BF16 mmproj at commit
   `1c04b8102ca5346f1faf4d9914503e378d713021`. Video follows images, after its own reference run. A quantised encoder
   is not used.
2. **The encoder's reference is llama.cpp's CPU backend** at ADR-0012's commit, through `tools/mtmd`, with flash
   attention off: its merged output rows per image (the golden embeddings) and `llama-mtmd-cli`'s greedy tokens. Its
   flash attention sums in F16 and moves some rows far from an exact computation. An image too large for the
   attention's scores in host memory takes an F32 computation of [MODEL_VISION.md](../spec/MODEL_VISION.md) instead,
   once that computation has matched llama.cpp on the smaller images
   ([the reference record](../measurements/MEASUREMENTS.md#2026-10-17-encoder-reference)).
3. **The image processing is the model's own** (`Qwen2VLImageProcessorFast` with the model's
   `preprocessor_config.json`, by Pillow), capped at the engine's largest image. The reference images are sized to a
   multiple of 32 pixels and inside both programs' limits, so neither resizes them.
4. **The encoder holds no VRAM of its own.** Its weights stay in one pinned host copy, streamed layer by layer into the
   prompt path's staging slots; its activations use the prompt path's regions. It runs between two prompt chunks,
   before the chunk that holds the image's tokens. When the staging slots are smaller than two of its layers, it
   allocates two of its own.
5. **The cards split an image by patches,** not by weights: each card runs every layer on half the patches, and they
   exchange keys and values once per layer. Each card copies its rows of K and V into the other card's memory, as
   [ADR-0030](0030-exchanges-into-the-other-card.md) allows, and events order the copies.
6. **Positions follow the model's M-RoPE.** Prompt rows carry three positions; a slot keeps the offset between its
   cells and its positions. For text the three are equal and the arithmetic is unchanged, bit for bit.
7. **The front end reads images sent in the request** (`data:` URLs, base64 sources). Fetching an `http(s)` URL is a
   start option, off by default.

## Why

The encoder is used by some requests some of the time; every expert evicted for it costs every token. Streaming its
weights costs about 1.3 ms a layer per card, hidden behind the layer's arithmetic from about half a megapixel up
(*estimate*). With one host copy and a split by patches, each card runs full-width products on half the rows, and a
layer exchanges 4.7 MB per card at 4,096 patches, against two exchanges of 9.4 MB had the weights been split
(*computed*).

## Options not chosen

- **The encoder resident in VRAM:** 439 experts per card for about 2-3% of decode speed, paid on every token.
- **Expert slots lent to the encoder,** with the experts copied back after: the staging slots are free at the same
  moment and need no copy back.
- **The weights split between the cards:** two exchanges a layer, half-width products, and the weights' bytes are not
  what bounds the encoder.
- **One card per image:** with one image the other card waits. Kept in case the exchange costs more than estimated.
- **The CPU runs the encoder with ggml:** seconds per image at a few TFLOPS (*estimate*), on the cores that compute
  the misses.
- **mtmd's image processing:** it pads instead of stretching and caps an image at 4,096 tokens; the model was trained
  with transformers' processor.

## Consequences

- The ABI gains version 3: `vision_path`, the encoder's limits in `engine_info`, and `engine_admit_images`.
- A prompt piece never ends inside a vision block; the largest image is at most `prompt_chunk - 2` tokens.
- Prefix checkpoints are keyed on the images' contents as well as the token ids.
- Text-only runs must show no change: `greedy_test`, and decode and prompt ABBA runs before and after.
- 0.90 GB more pinned host memory, from the arena's pool ([MEMORY_BUDGET.md](../measurements/MEMORY_BUDGET.md)).
