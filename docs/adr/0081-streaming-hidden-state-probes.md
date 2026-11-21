# ADR-0081: streaming hidden-state probes opt in through the moderation API

**Status:** draft. 2026-11-21.
**Revisit when:** an ABBA run against `main` shows a slowdown when `moderation` is absent, probing adds more than 2% wall time when active, or a probe architecture needs more than 512 KiB of KV state per slot.

## Context

Autonomous assistants need checks for unsafe prompts, indirect prompt injection in tool output, unsafe replies and hallucinations while tokens stream. Running a separate guardrail model duplicates the forward pass and holds a second set of weights in memory.

Intermediate hidden states of Qwen3.8-Flash-Next already exist during prefill and decode on card 0. `inclusionAI/Qwen3.8-Flash-Next-singprobe` is a 5.18M-parameter single-layer multi-query attention probe (10.4 MB in BF16) that taps layers 15, 31 and 47 of Qwen3.8-Flash-Next, keeps a 2,048-token sliding window (512 KiB of BF16 K and V per slot), and predicts ten labels across input risk categories, `safe`, `unsafe_response` and `hallucination`. Linear probes and before-and-after span-drift detectors (TaskTracker style) read the same tapped layers.

Requests that do not ask for moderation must pay no kernel launch, no extra copy and no memory bandwidth on the hot decode and prompt paths ([ADR-0016](0016-how-a-change-is-judged.md)).

## Decision

1. **Opt in through standard moderation fields.** Probing is off by default. `POST /v1/chat/completions` and `POST /v1/messages` enable it per request via a `moderation` object (`model`, `policy.input.mode` and `policy.output.mode` in `score` or `block`, and optional `untrusted_spans`). Responses include `moderation` only when the request asked for it. `POST /v1/moderations` runs prefill-only scoring with zero generated tokens (`ENGINE_PROBE_ONLY`).
2. **Separate command lists keep unprobed steps unchanged.** The normal decode and prompt command lists hold no probe kernels. Probing uses separate lazily recorded decode and prompt command lists (`probe_tap` and `p_probe_tap`) invoked only when an admitted slot in the step has probing enabled.
3. **Commit accepted tokens only.** In speculative decode rounds (`round` and `rounds`), the engine runs `probe_commit_step` only on accepted rows. Rejected draft tokens never enter the probe KV ring or rolling-mean tracker. When `block` mode trips on input or output, generation stops with `ENGINE_DONE_CONTENT_FILTER` (`finish_reason: "content_filter"` on OpenAI, `stop_reason: "refusal"` on Anthropic).
4. **Probe checkpoints match probed requests.** Slot checkpoints include the 512 KiB probe KV ring and rolling-mean tracker state when taken from a probed slot. A probed request restores only checkpoints that carry probe state and do not split an untrusted span. An unprobed request may restore any checkpoint.
5. **Three probe formats and an optional calibration file.** The core loads `sing_probe_attn`, `sing_probe_mlp` and `c0ven_probe_v1` (`config.json` and `model.safetensors`), plus an optional `calibration.json` holding Platt scale and bias, per-label thresholds, `roll_window` and an optional `extra_alarms` linear or drift sidecar sharing the same tapped layers. The engine auto-detects `singprobe/` beside the model file unless `--no-probe` or `--probe <dir>` overrides it.
6. **C ABI version 24.** `engine_config` adds `probe_path`, `engine_info` adds `probe_labels`, `engine_settings` adds `probe_flags` and `probe_spans`, `engine_done_reason` adds `ENGINE_DONE_CONTENT_FILTER`, and `engine_read_probe_info` and `engine_read_slot_probe` read probe metadata and per-slot scores.

## Options not chosen

- **Run the probe on every request unconditionally.** Adds tap kernels and head launches to traffic that never reads moderation output, risking regression on the unprobed decode and prefill benchmarks.
- **Keep a custom non-standard API field.** OpenAI SDKs already define `POST /v1/moderations` and the `moderation` object on Chat Completions, and Anthropic defines `stop_reason: "refusal"`. Matching those schemas avoids client glue.
- **Evaluate the probe inside the speculative forward pass.** Draft tokens are frequently rejected and would corrupt the probe sliding window and rolling mean unless rolled back on card 0.

## Consequences

- Unprobed traffic executes the unmodified command lists and pays zero GPU overhead.
- Probed decode steps add one `probe_tap` per tapped layer and one `probe_mqa_head` launch on card 0 (*estimate*: under 0.15 ms per decode step, from the 5.18M parameter count at B60 BF16 XMX throughput and 512 KiB ring read).
- Each slot takes 512 KiB of card-0 VRAM for the `sing_probe_attn` ring, accounted for in `engine/models/qwen4exp/probe.py` and in `checkpoint_bytes`.
- `tools/probe_calibrate.py` calibrates thresholds and Platt parameters against labeled data and fits `c0ven_probe_v1` linear or drift probes from `.npz` hidden-state arrays.
