# API

The server speaks the OpenAI and Anthropic APIs, with streaming, on the address and port in its
[options](OPTIONS.md).

| Endpoint | What it is |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
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
