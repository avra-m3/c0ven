"""Cached-prefix multiple-choice scoring: `POST /v1/decide`.

Reads a conversation prefix (reusing any matching checkpoint without saving a new one) plus a short check prompt,
gathers exact raw logits for up to 16 answer choice tokens on the final prompt position, and returns calibrated
probabilities, full-vocabulary probabilities and optional position-debiased swapped-order averages.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections.abc import Coroutine, Sequence
from contextlib import aclosing
from dataclasses import dataclass, replace
from typing import Any, Protocol

from tokenizers import Tokenizer

import protocol
from frontend import generate, openai_api
from frontend.body import array, boolean, number, obj
from frontend.generate import BadRequest, End, Engine, Finish
from frontend.model import Model
from frontend.template import Conversation, Efforts
from frontend.tokenizer import encode

type JSONObject = dict[str, Any]

AUTO_KEYS = tuple(str(i) for i in range(1, 10)) + tuple("ABCDEFG")
"""Single-character keys assigned when `check_prompt` is given with plain choice labels."""


@dataclass(frozen=True, slots=True)
class ChoiceOption:
    label: str
    description: str
    key: str
    token_id: int


@dataclass(frozen=True, slots=True)
class DecideRequest:
    conversation: Conversation
    choices: tuple[ChoiceOption, ...]
    check_prompt: str | None = None
    temperature: float = 1.0
    swap_order: bool = False
    close_thinking: bool = True
    assistant_prefix: str = ""
    auto_keys: bool = False
    images: tuple[bytes, ...] = ()
    videos: tuple[bytes, ...] = ()


@dataclass(frozen=True, slots=True)
class PassResult:
    order: tuple[str, ...]
    logits: dict[str, float]
    probabilities: dict[str, float]
    full_probabilities: dict[str, float]
    logsumexp: float
    end: End


def calibrated_probs(logits: Sequence[float], temperature: float) -> list[float]:
    """Calibrated softmax over `logits` divided by `temperature` (> 0). Numerically stable across any dynamic range."""
    scaled = [v / temperature for v in logits]
    mx = max(scaled)
    exps = [math.exp(v - mx) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def _single_token(tokenizer: Tokenizer, text: str, what: str) -> int:
    ids = encode(tokenizer, text)
    if len(ids) != 1:
        raise BadRequest(f"{what}: {text!r} encodes to {len(ids)} tokens, must be 1 (or pass token_id)")
    return ids[0]


def _parse_choices(raw: list[Any], tokenizer: Tokenizer, has_check: bool) -> tuple[tuple[ChoiceOption, ...], bool]:
    if not 1 <= len(raw) <= protocol.MAX_SCORE_TOKENS:
        raise BadRequest(f"choices must hold 1..{protocol.MAX_SCORE_TOKENS} items")
    vocab_size = tokenizer.get_vocab_size()
    use_auto = has_check and all(
        isinstance(item, str) or (isinstance(item, dict) and "token" not in item and "token_id" not in item)
        for item in raw
    )
    out: list[ChoiceOption] = []
    seen_labels: set[str] = set()
    seen_ids: set[int] = set()
    for i, item in enumerate(raw):
        what = f"choices[{i}]"
        if isinstance(item, str):
            if not item:
                raise BadRequest(f"{what} must be non-empty")
            label = description = item
            key = AUTO_KEYS[i] if use_auto else item
            token_id = _single_token(tokenizer, key, what)
        else:
            d = obj(item, what)
            label_v = d.get("label")
            if not isinstance(label_v, str) or not label_v:
                raise BadRequest(f"{what}.label must be a non-empty string")
            label = label_v
            desc_v = d.get("description", label)
            if not isinstance(desc_v, str) or not desc_v:
                raise BadRequest(f"{what}.description must be a non-empty string")
            description = desc_v
            if "token_id" in d and d["token_id"] is not None:
                tid = d["token_id"]
                if isinstance(tid, bool) or not isinstance(tid, int) or not 0 <= tid < vocab_size:
                    raise BadRequest(f"{what}.token_id must be an integer in 0..{vocab_size - 1}")
                token_id = tid
                key_v = d.get("token")
                key = key_v if isinstance(key_v, str) and key_v else tokenizer.decode([tid], skip_special_tokens=False)
            elif "token" in d and d["token"] is not None:
                tok_v = d["token"]
                if not isinstance(tok_v, str) or not tok_v:
                    raise BadRequest(f"{what}.token must be a non-empty string")
                key = tok_v
                token_id = _single_token(tokenizer, key, f"{what}.token")
            elif use_auto:
                key = AUTO_KEYS[i]
                token_id = _single_token(tokenizer, key, what)
            else:
                raise BadRequest(f"{what} must give token or token_id")
        if label in seen_labels:
            raise BadRequest(f"duplicate choice label {label!r}")
        if token_id in seen_ids:
            raise BadRequest(f"duplicate choice token id {token_id}")
        seen_labels.add(label)
        seen_ids.add(token_id)
        out.append(ChoiceOption(label, description, key, token_id))
    return tuple(out), use_auto


def parse(body: JSONObject, tokenizer: Tokenizer, efforts: Efforts) -> DecideRequest:
    """Validate and parse a `POST /v1/decide` JSON body."""
    conv, _, images, videos = openai_api.conversation_from(body, efforts)
    check_v = body.get("check_prompt")
    check_prompt: str | None = None
    if check_v is not None:
        if not isinstance(check_v, str) or not check_v.strip():
            raise BadRequest("check_prompt must be a non-empty string")
        check_prompt = check_v
    choices, auto_keys = _parse_choices(array(body.get("choices"), "choices"), tokenizer, check_prompt is not None)
    temp_v = number(body, "temperature", float)
    temperature = 1.0 if temp_v is None else float(temp_v)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise BadRequest("temperature must be a finite number > 0")
    swap_v = body.get("swap_order", False)
    if not isinstance(swap_v, bool):
        raise BadRequest("swap_order must be a boolean")
    if swap_v:
        if check_prompt is None:
            raise BadRequest("swap_order requires check_prompt")
        if len(choices) < 2:
            raise BadRequest("swap_order requires at least 2 choices")
    close_v = body.get("close_thinking", True)
    close_thinking = boolean(close_v)
    if close_thinking is None:
        raise BadRequest("close_thinking must be a boolean")
    prefix_v = body.get("assistant_prefix", "")
    if not isinstance(prefix_v, str):
        raise BadRequest("assistant_prefix must be a string")
    return DecideRequest(
        conversation=conv,
        choices=choices,
        check_prompt=check_prompt,
        temperature=temperature,
        swap_order=swap_v,
        close_thinking=close_thinking,
        assistant_prefix=prefix_v,
        auto_keys=auto_keys,
        images=images,
        videos=videos,
    )


def _format_check(check_prompt: str, keys: Sequence[str], ordered: Sequence[ChoiceOption]) -> str:
    lines = [check_prompt, *(f"{key}: {opt.description}" for key, opt in zip(keys, ordered, strict=True))]
    lines.append("Answer with the choice key only:")
    return "\n".join(lines)


def _pass_conversation(req: DecideRequest, ordered: Sequence[ChoiceOption]) -> Conversation:
    if req.check_prompt is None:
        return req.conversation
    keys = [c.key for c in (req.choices if req.auto_keys else ordered)]
    extra: JSONObject = {"role": "user", "content": _format_check(req.check_prompt, keys, ordered)}
    return replace(req.conversation, messages=[*req.conversation.messages, extra])


class RenderFn(Protocol):
    def __call__(
        self,
        req: generate.Request,
        *,
        close_thinking: bool = False,
        assistant_prefix: str = "",
    ) -> Coroutine[Any, Any, generate.Prompt]: ...


async def _run_pass(
    engine: Engine,
    req: DecideRequest,
    ordered: Sequence[ChoiceOption],
    prompt: generate.Prompt,
) -> PassResult:
    if len(prompt.ids) > engine.max_context:
        raise BadRequest(f"the prompt ({len(prompt.ids)} tokens) exceeds the context ({engine.max_context})")
    # Auto-assigned positional keys stay fixed across passes; caller-bound tokens follow their choice.
    token_ids = tuple(c.token_id for c in (req.choices if req.auto_keys else ordered))
    labels = tuple(c.label for c in ordered)
    scores: protocol.Scores | None = None
    cached = 0
    end: End | None = None
    async with aclosing(engine.decide(prompt.ids, token_ids, prompt.images)) as events:
        async for event in events:
            match event:
                case protocol.Resume(reused=reused):
                    cached = reused
                case protocol.Scores() as sc:
                    scores = sc
                case protocol.Done(reused=reused, prompt_ms=prompt_ms, queue_ms=queue_ms) as done:
                    cached = reused or cached
                    end = End(
                        Finish.STOP,
                        len(prompt.ids),
                        0,
                        cached,
                        prompt_ms=prompt_ms,
                        decode_ms=0.0,
                        queue_ms=queue_ms,
                        steps=done.steps,
                    )
                    break
                case protocol.Error(message=message):
                    end = End(Finish.ERROR, len(prompt.ids), 0, cached, error=message)
                    break
                case _:
                    pass
    if end is None:
        end = End(Finish.ERROR, len(prompt.ids), 0, cached, error="the engine ended before scoring finished")
    if end.error is not None:
        return PassResult(labels, {}, {}, {}, 0.0, end)
    if scores is None or len(scores.logits) != len(labels):
        return PassResult(
            labels,
            {},
            {},
            {},
            0.0,
            replace(end, finish=Finish.ERROR, error="the engine returned no choice scores"),
        )
    probs = calibrated_probs(scores.logits, req.temperature)
    full = [math.exp(v - scores.logsumexp) for v in scores.logits]
    return PassResult(
        order=labels,
        logits=dict(zip(labels, scores.logits, strict=True)),
        probabilities=dict(zip(labels, probs, strict=True)),
        full_probabilities=dict(zip(labels, full, strict=True)),
        logsumexp=scores.logsumexp,
        end=end,
    )


async def execute(
    model: Model,
    engine: Engine,
    req: DecideRequest,
    render_fn: RenderFn,
) -> tuple[JSONObject | None, End]:
    """Run one check pass (or two parallel swapped-order passes) and return the response body and summary End."""
    orders: list[tuple[ChoiceOption, ...]] = [req.choices]
    if req.swap_order:
        orders.append(tuple(reversed(req.choices)))

    async def one(ordered: tuple[ChoiceOption, ...]) -> PassResult:
        conv = _pass_conversation(req, ordered)
        gen_req = generate.Request(conv, model.defaults, images=req.images, videos=req.videos)
        prompt = await render_fn(gen_req, close_thinking=req.close_thinking, assistant_prefix=req.assistant_prefix)
        return await _run_pass(engine, req, ordered, prompt)

    results = list(await asyncio.gather(*(one(order) for order in orders)))
    for res in results:
        if res.end.error is not None:
            return None, res.end

    labels = [c.label for c in req.choices]
    n = len(results)
    probs = {label: sum(r.probabilities[label] for r in results) / n for label in labels}
    logits = {label: sum(r.logits[label] for r in results) / n for label in labels}
    full = {label: sum(r.full_probabilities[label] for r in results) / n for label in labels}
    best = max(labels, key=lambda label: probs[label])

    prompt_tokens = sum(r.end.prompt_tokens for r in results)
    cached_tokens = sum(r.end.cached_tokens for r in results)
    prompt_ms = max(r.end.prompt_ms for r in results)
    queue_ms = max(r.end.queue_ms for r in results)
    summary = End(
        Finish.STOP,
        prompt_tokens,
        0,
        cached_tokens,
        prompt_ms=prompt_ms,
        decode_ms=0.0,
        queue_ms=queue_ms,
    )
    body: JSONObject = {
        "id": f"dec-{uuid.uuid4().hex}",
        "object": "decision",
        "created": int(time.time()),
        "model": model.name,
        "choice": best,
        "probabilities": probs,
        "logits": logits,
        "full_probabilities": full,
        "mass": sum(full.values()),
        "temperature": req.temperature,
        "swapped": req.swap_order,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
            "prompt_ms": round(prompt_ms, 3),
        },
    }
    if req.swap_order:
        body["passes"] = [
            {
                "order": list(r.order),
                "probabilities": r.probabilities,
                "logits": r.logits,
                "full_probabilities": r.full_probabilities,
            }
            for r in results
        ]
    return body, summary
