# ADR-0034: video in scope, as pairs of frames through the image encoder

**Status:** draft, 2026-10-17; not accepted. Builds on [ADR-0031](0031-images-through-the-prompt-path.md), also a
draft, whose item 1 leaves video for after its own reference run.
**Revisit when:** a video's pairs measure too slow to encode between prompt chunks, the prompt format the model was
trained with turns out to differ from transformers v4.57.0's, or llama.cpp's video path becomes the model's own.

## Context

- **The model reads video.** Its repository ships `video_preprocessor_config.json` for `Qwen3VLVideoProcessor`, and
  its chat template renders a video item. How a video becomes frames, pairs, patches and prompt text is
  [MODEL_VISION.md, section 6](../spec/MODEL_VISION.md#6-video).
- **A pair of frames is an image to the encoder.** Each pair of consecutive sampled frames is one temporal patch. Its
  attention, position table and rotation are those of an image of the same grid; only the patch embedding differs,
  `W0 p0 + W1 p1` in place of `(W0 + W1) p`. transformers and llama.cpp agree on that
  ([MODEL_VISION.md, sections 6.3 and 6.4](../spec/MODEL_VISION.md#63-a-pairs-patches)).
- **The reference agrees with the spec.** llama.cpp's CPU encoder on two frame pairs of its own `test-3.mp4`, with
  flash attention off, matches an F64 computation of the spec to a least cosine of 0.9966 at 252 tokens; with the two
  frames swapped the least cosine falls to 0.34
  ([the video reference record](../measurements/MEASUREMENTS.md#2026-10-17-video-pair-reference)).
- **The sources differ on the prompt.** transformers at the pinned `main` doubles the vision marks around a video;
  v4.57.0's processor does not. llama.cpp's video helper uses its own frame rate and time format. The n-gram token at
  a video's positions is 248,056 in llama.cpp and the prompt's 248,057 in transformers
  ([MODEL_VISION.md, section 9](../spec/MODEL_VISION.md#9-where-the-sources-differ), items 4-6).
- **The front end has no video decoder.** Pillow reads still images only.

## Decision

1. **Video is in scope** for Qwen3.8-Flash-Next, through ADR-0031's encoder, weights and buffers. Nothing is added to
   VRAM but the two frames' patch kernels kept apart beside their sum, 7.1 MB per card.
2. **A video is a sequence of frame pairs, each a block of its own.** The engine's image record gains a frame count,
   1 or 2. A pair's patches carry both frames, the earlier first; everything after the patch embedding is the image
   path unchanged. The ABI takes it as `engine_image.frames`, version 6.
3. **The reference for a pair's rows is llama.cpp's CPU encoder** at ADR-0012's commit, a batch of two frames, flash
   attention off, as ADR-0031, item 2, has it for images.
4. **Frames are sampled and sized by the model's processor's rules**, with qwen-vl-utils' per-frame cap on: 2 frames
   a second, 4 to 768 frames, at most 768 tokens' worth of pixels a frame, then the engine's `image_tokens_max` per
   pair.
5. **The prompt text is transformers v4.57.0's**: per pair, `<t seconds>` and the pair's own vision marks, replacing
   the template's three tokens. A pair's block holds `<|video_pad|>`, and the n-gram hash reads it as it is.
6. **The front end decodes video with PyAV** 19.0.1 (BSD-3-Clause; its wheel carries FFmpeg 9.0.2's libraries),
   pinned in `uv.lock` as ADR-0004 requires. It reads `video_url` parts with `data:` URLs in the OpenAI API, as vLLM's
   server names them, and a `video` block with a base64 source in the Anthropic API. Fetching stays off (ADR-0031,
   item 7).

## Why

Pairs reuse what ADR-0031 built and measured: the encoder needs one new argument and one more matrix per card, and a
pair's cost is an image's of the same grid. A video's tokens are bounded the way an image's are, by the context and
by `image_tokens_max` per pair, so no new limit reaches the scheduler.

The prompt follows the model's processor for the reason ADR-0031, item 3, gives for images: the model was trained
with it. Of transformers' two versions, v4.57.0's replaces the template's three tokens as one, so each pair has one
pair of marks. At `main` the replacement is the generic one for every model, which finds only the pad token and
leaves the template's marks around the whole video.

## Options not chosen

- **All of a video's frames in one attention span.** Neither source does it; the model's attention is per pair.
- **llama.cpp's video helper as the prompt's reference.** Its 4 frames a second and `[0m0.00s]` times every 5
  seconds are not the model's format.
- **The image token, 248,056, at a video's positions,** as llama.cpp hashes them. The engine's prompt then would not
  be the template's, and the model's own code hashes 248,057.
- **ffmpeg run as a program.** Another binary in the image, and frames passed through a pipe; PyAV is one pinned
  wheel.
- **No per-frame cap,** transformers' current default. Short clips would then take nearly the whole 25,165,824
  pixels; transformers warns that the cap becomes its default.

## Consequences

- The ABI goes to version 6: `engine_image.reserved` becomes `frames`.
- The IMG line carries the frame count, and a video is one IMG line per pair.
- The encoder's pixel buffers hold two frames a patch, on the host and in the prompt regions, which can lower
  `image_tokens_max` by a few windows.
- encoder_test takes frame-pair references beside the images; vision_test checks `v_embed` with two frames.
- `pyproject.toml` gains `av`; the front end's image needs no system FFmpeg.
