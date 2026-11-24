"""Standard OpenAI moderation request parsing, tool-span extraction, and result formatting."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import protocol
from frontend.body import array, obj
from frontend.generate import BadRequest, Prompt
from frontend.model import Model
from frontend.template import Conversation, TemplateFailed
from frontend.tokenizer import encode

type JSONObject = dict[str, Any]

_MODES = ("score", "block")
_INPUT_SCOPES = ("input", "both", "drift")
_OUTPUT_SCOPES = ("output", "both")


@dataclass(frozen=True, slots=True)
class ModerationRequest:
    """Per-request opt-in streaming probe configuration from a `moderation` JSON object."""

    model: str | None = None
    block_input: bool = False
    block_output: bool = False
    untrusted_spans: tuple[tuple[int, int], ...] | None = None

    def flags(self) -> int:
        out = protocol.PROBE_ENABLED
        if self.block_input:
            out |= protocol.PROBE_BLOCK_INPUT
        if self.block_output:
            out |= protocol.PROBE_BLOCK_OUTPUT
        return out


def parse_request(raw: object) -> ModerationRequest | None:
    """Parse an optional `moderation` object on Chat Completions or Messages; None disables probing."""
    if raw is None:
        return None
    d = obj(raw, "moderation")
    model_val = d.get("model")
    if model_val is not None and (not isinstance(model_val, str) or not model_val):
        raise BadRequest("moderation.model must be a non-empty string")
    block_input = False
    block_output = False
    if (policy_raw := d.get("policy")) is not None:
        policy = obj(policy_raw, "moderation.policy")
        for key in policy:
            if key not in ("input", "output"):
                raise BadRequest(f"moderation.policy: unknown key {key!r}")
        for side in ("input", "output"):
            if (side_raw := policy.get(side)) is not None:
                side_obj = obj(side_raw, f"moderation.policy.{side}")
                mode = side_obj.get("mode", "score")
                if mode not in _MODES:
                    raise BadRequest(f"moderation.policy.{side}.mode must be 'score' or 'block', not {mode!r}")
                if side == "input":
                    block_input = mode == "block"
                else:
                    block_output = mode == "block"
    spans: tuple[tuple[int, int], ...] | None = None
    if (spans_raw := d.get("untrusted_spans")) is not None:
        parsed: list[tuple[int, int]] = []
        for i, item in enumerate(array(spans_raw, "moderation.untrusted_spans")):
            sp = obj(item, f"moderation.untrusted_spans[{i}]")
            start, end = sp.get("start"), sp.get("end")
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
            ):
                raise BadRequest(f"moderation.untrusted_spans[{i}] needs integer start and end")
            if start < 0 or start >= end:
                raise BadRequest(f"moderation.untrusted_spans[{i}] must satisfy 0 <= start < end")
            parsed.append((start, end))
        if len(parsed) > protocol.MAX_PROBE_SPANS:
            raise BadRequest(
                f"moderation.untrusted_spans has {len(parsed)} entries, at most {protocol.MAX_PROBE_SPANS}"
            )
        spans = tuple(parsed)
    return ModerationRequest(model=model_val, block_input=block_input, block_output=block_output, untrusted_spans=spans)


def validate(req: ModerationRequest, meta: protocol.ProbeMeta | None, prompt_len: int) -> protocol.ProbeMeta:
    """Verify the loaded probe satisfies `req` and spans lie inside `prompt_len`."""
    if meta is None:
        raise BadRequest("moderation requested, but no streaming probe is loaded")
    if req.model is not None and req.model != meta.model:
        raise BadRequest(f"unknown moderation model {req.model!r}; loaded probe is {meta.model!r}")
    if req.untrusted_spans is not None:
        for start, end in req.untrusted_spans:
            if end > prompt_len:
                raise BadRequest(f"moderation untrusted_span {start}:{end} exceeds prompt length {prompt_len}")
    return meta


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


def _adjust_for_images(tok: int, prompt: Prompt) -> int:
    if not prompt.images:
        return tok
    shift = 0
    for im in prompt.images:
        pre = im.first - shift
        extra = im.tokens - 1
        if tok > pre:
            tok += extra
        shift += extra
    return tok


def detect_tool_spans(model: Model, conversation: Conversation, prompt: Prompt) -> tuple[tuple[int, int], ...]:
    """Compute [start, end) prompt token spans covering every `role == 'tool'` message."""
    indices = [i for i, m in enumerate(conversation.messages) if m.get("role") == "tool"]
    if not indices:
        return ()
    spans: list[tuple[int, int]] = []
    n_ids = len(prompt.ids)
    for idx in indices:
        try:
            before = (
                model.template.render(
                    replace(conversation, messages=conversation.messages[:idx], add_generation_prompt=False)
                )
                if idx > 0
                else ""
            )
            after = model.template.render(
                replace(conversation, messages=conversation.messages[: idx + 1], add_generation_prompt=False)
            )
        except TemplateFailed:
            continue
        start_c = _common_prefix_len(before, prompt.text)
        end_c = _common_prefix_len(after, prompt.text)
        if end_c <= start_c:
            continue
        start_t = _adjust_for_images(len(encode(model.tokenizer, prompt.text[:start_c])), prompt)
        end_t = _adjust_for_images(len(encode(model.tokenizer, prompt.text[:end_c])), prompt)
        start_t = max(0, min(start_t, n_ids))
        end_t = max(0, min(end_t, n_ids))
        if start_t < end_t:
            spans.append((start_t, end_t))
    return tuple(spans[-protocol.MAX_PROBE_SPANS :])


def _side_entry(
    meta: protocol.ProbeMeta, scores: Sequence[float], flagged: bool, scopes: tuple[str, ...]
) -> JSONObject | None:
    categories: dict[str, bool] = {}
    category_scores: dict[str, float] = {}
    for lb, raw in zip(meta.labels, scores, strict=True):
        if lb.safe or lb.scope not in scopes:
            continue
        val = float(raw)
        categories[lb.name] = val >= lb.threshold
        category_scores[lb.name] = val
    if not categories:
        return None
    is_flagged = bool(flagged or any(categories.values()))
    return {
        "type": "moderation_results",
        "model": meta.model,
        "results": [
            {
                "type": "moderation_result",
                "model": meta.model,
                "flagged": is_flagged,
                "categories": categories,
                "category_scores": category_scores,
                "category_applied_input_types": {name: ["text"] for name in categories},
            }
        ],
    }


def format_chat(meta: protocol.ProbeMeta, report: protocol.ProbeReport, *, include_output: bool = True) -> JSONObject:
    """Format a ProbeReport as an OpenAI `ChatCompletion.moderation` object."""
    out: JSONObject = {}
    inp = _side_entry(meta, report.input_scores, report.input_flagged, _INPUT_SCOPES)
    if inp is not None:
        out["input"] = inp
    if include_output:
        outp = _side_entry(meta, report.output_scores, report.output_flagged, _OUTPUT_SCOPES)
        if outp is not None:
            out["output"] = outp
    return out


def parse_moderations(body: JSONObject, meta: protocol.ProbeMeta | None) -> tuple[list[str], protocol.ProbeMeta]:
    """Validate a `POST /v1/moderations` body and return the input strings and loaded probe metadata."""
    if meta is None:
        raise BadRequest("no streaming probe is loaded")
    model_val = body.get("model")
    if model_val is not None:
        if not isinstance(model_val, str) or not model_val:
            raise BadRequest("model must be a non-empty string")
        if model_val != meta.model:
            raise BadRequest(f"unknown moderation model {model_val!r}; loaded probe is {meta.model!r}")
    raw = body.get("input")
    if isinstance(raw, str):
        if not raw:
            raise BadRequest("input must not be empty")
        return [raw], meta
    items = array(raw, "input")
    if not items:
        raise BadRequest("input must not be empty")
    texts: list[str] = []
    for i, item in enumerate(items):
        if not isinstance(item, str) or not item:
            raise BadRequest(f"input[{i}] must be a non-empty string")
        texts.append(item)
    return texts, meta


def format_moderations_response(meta: protocol.ProbeMeta, reports: Sequence[protocol.ProbeReport]) -> JSONObject:
    """Format a batch of input ProbeReports as a standard OpenAI `/v1/moderations` response."""
    results: list[JSONObject] = []
    has_input_labels = any(not lb.safe and lb.scope in _INPUT_SCOPES for lb in meta.labels)
    scopes = _INPUT_SCOPES if has_input_labels else _INPUT_SCOPES + _OUTPUT_SCOPES
    for report in reports:
        categories: dict[str, bool] = {}
        category_scores: dict[str, float] = {}
        for lb, raw in zip(meta.labels, report.input_scores, strict=True):
            if lb.safe or lb.scope not in scopes:
                continue
            val = float(raw)
            categories[lb.name] = val >= lb.threshold
            category_scores[lb.name] = val
        results.append(
            {
                "flagged": bool(report.input_flagged or any(categories.values())),
                "categories": categories,
                "category_scores": category_scores,
                "category_applied_input_types": {name: ["text"] for name in categories},
            }
        )
    return {"id": f"modr-{uuid.uuid4().hex}", "model": meta.model, "results": results}


def props_summary(meta: protocol.ProbeMeta | None) -> JSONObject | None:
    """JSON summary of the loaded streaming probe for `GET /props`, or None."""
    if meta is None:
        return None
    return {
        "model": meta.model,
        "kind": meta.kind,
        "window": meta.window,
        "roll_window": meta.roll_window,
        "labels": [
            {"name": lb.name, "scope": lb.scope, "threshold": lb.threshold, "safe": lb.safe} for lb in meta.labels
        ],
    }
