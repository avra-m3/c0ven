# API

The server speaks the OpenAI and Anthropic APIs, with streaming, on the address and port in its
[options](OPTIONS.md).

| Endpoint | What it is |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/decide` | Cached-prefix choice scoring over up to 16 answer tokens |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages |
| `GET /v1/models` | The loaded model |
| `GET /health` | 200 and the engine's context when it is ready; 503 while it loads or sleeps |
| `GET /props` | Chat template, context, generation defaults |
| `GET /slots` | Slots and how many are busy |
| `GET /metrics` | Engine and front end counters, as JSON: requests, the core's steps, misses, drafts and swaps, and the ratios read from them |
| `POST /admin/wake`, `/admin/sleep`, `/admin/lease` | On `--admin-port` only: load the model, give the cards back, and hold it awake ([Share the cards](guides/share-the-cards.md)) |

## Images and video

Images and videos are read from `data:` URLs or base64 only, never fetched from a URL: `image_url` and `video_url`
parts in the OpenAI API, `image` and `video` blocks in the Anthropic API. The designs,
[ADR-0031](adr/0031-images-through-the-prompt-path.md) for images and
[ADR-0034](adr/0034-video-as-pairs-of-frames.md) for video, are drafts, not yet accepted. The server needs the
[image encoder file](MODEL_FILES.md#the-image-encoder) for them.

## Choice scoring

`POST /v1/decide` evaluates a short multiple-choice check against a conversation prefix without generating text or
saving new checkpoints ([ADR-0080](adr/0080-cached-prefix-choice-scoring.md)). It accepts OpenAI-shaped `messages`,
optional `tools`, `chat_template_kwargs`, and `reasoning_effort`, plus `choices` (1 to 16 single-token strings, labels
paired with `check_prompt`, or objects with `label` and `token` or `token_id`).

Optional fields:

- `check_prompt`: appends a user turn listing each choice with its key so the prefix before it matches saved
  checkpoints.
- `temperature`: positive scaling divisor applied before softmax across `choices` (default `1.0`).
- `swap_order`: with `check_prompt` and at least two choices, runs forward and reversed choice orderings in parallel
  off the same checkpoint and averages their probabilities.
- `close_thinking`: when `true` (default) and the template ends inside `<think>`, closes the block at the tail without
  changing the leading prefix.
- `assistant_prefix`: literal text appended right before scoring.

The response returns the top `choice`, calibrated `probabilities` summing to 1, raw `logits`, full-vocabulary
`full_probabilities` and their sum `mass`, `usage` (including `cached_tokens`), and `passes` when `swap_order` is
true.
