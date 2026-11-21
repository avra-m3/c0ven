# API

The server speaks the OpenAI and Anthropic APIs, with streaming, on the address and port in its
[options](OPTIONS.md).

| Endpoint | What it is |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/moderations` | OpenAI Moderations using the loaded streaming hidden-state probe (prefill only) |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages |
| `GET /v1/models` | The loaded model |
| `GET /health` | 200 and the engine's context when it is ready; 503 while it loads or sleeps |
| `GET /props` | Chat template, context, generation defaults, loaded probe summary |
| `GET /slots` | Slots and how many are busy |
| `GET /metrics` | Engine and front end counters, as JSON: requests, the core's steps, misses, drafts and swaps, and the ratios read from them |
| `POST /admin/wake`, `/admin/sleep`, `/admin/lease` | On `--admin-port` only: load the model, give the cards back, and hold it awake ([Share the cards](guides/share-the-cards.md)) |

## Images and video

Images and videos are read from `data:` URLs or base64 only, never fetched from a URL: `image_url` and `video_url`
parts in the OpenAI API, `image` and `video` blocks in the Anthropic API. The designs,
[ADR-0031](adr/0031-images-through-the-prompt-path.md) for images and
[ADR-0034](adr/0034-video-as-pairs-of-frames.md) for video, are drafts, not yet accepted. The server needs the
[image encoder file](MODEL_FILES.md#the-image-encoder) for them.

## Moderation and streaming probes

When a [streaming hidden-state probe](MODEL_FILES.md#the-streaming-probe) is loaded
([ADR-0081](adr/0081-streaming-hidden-state-probes.md)), `GET /props` lists its model name and labels under `probe`.
Probing is off unless a request opts in.

On `POST /v1/chat/completions` and `POST /v1/messages`, pass a `moderation` object:

```json
{
  "moderation": {
    "model": "Qwen3.8-Flash-Next-singprobe",
    "policy": {
      "input": { "mode": "block" },
      "output": { "mode": "score" }
    }
  }
}
```

All keys inside `moderation` are optional (`{}` scores both input and output). `mode` is `score` (default) or `block`.
In `block` mode, a threshold hit stops generation with `finish_reason: "content_filter"` on OpenAI and
`stop_reason: "refusal"` on Anthropic. For drift probes, the server automatically marks every `tool` message as an
untrusted span unless the request passes `untrusted_spans: [{"start": ..., "end": ...}]`.

Non-streaming responses include a top-level `moderation` object with `input` and `output` `moderation_results` entries.
Streaming responses deliver `moderation` in the trailing `choices: []` chunk before `[DONE]` (OpenAI) or on
`message_delta` (Anthropic). Requests without `moderation` omit the field and execute without probe kernels.

`POST /v1/moderations` accepts `{"input": "..." | ["...", ...], "model": "..."}` and scores each input during prefill
without generating tokens.
