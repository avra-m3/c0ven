"""OpenAI Chat Completions: `POST /v1/chat/completions`, streamed and not, with tools and JSON output.

Beside OpenAI's own fields it takes the ones llama.cpp's and vLLM's servers take: `top_k`, `min_p`,
`repetition_penalty`, `reasoning_content` on assistant messages, and `chat_template_kwargs` (`enable_thinking`,
`reasoning_effort`, `preserve_thinking`). OpenAI's `reasoning_effort` maps onto the template's three levels. A user
message's `image_url` parts are read when they are `data:` URLs (frontend/images.py), and so are its `video_url` parts
(frontend/video.py), as vLLM's server names them.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import protocol
from frontend import video
from frontend.body import array, boolean, number, obj, settings
from frontend.generate import BadRequest, End, Finish, Request, Result
from frontend.images import ImageError, from_data_url
from frontend.output import Content, Piece, Reasoning, ToolCall, ToolCallStart
from frontend.template import Conversation

type JSONObject = dict[str, Any]

EFFORTS = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}
JSON_INSTRUCTION = "Answer with a single JSON value only, with no other text."

_FINISH = {
    Finish.STOP: "stop",
    Finish.STOP_STRING: "stop",
    Finish.LENGTH: "length",
    Finish.TOOL_CALLS: "tool_calls",
    Finish.CANCEL: "stop",
    Finish.ERROR: "error",
}


def _text(content: object, what: str) -> str:
    """A message's content: a string, or text parts. Pictures are refused here: only user messages carry them."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in array(content, what):
        p = obj(part, what)
        if p.get("type") != "text":
            raise BadRequest(f"{what}: only text parts are supported, not {p.get('type')!r}")
        parts.append(str(p.get("text", "")))
    return "".join(parts)


def _url(part: JSONObject, key: str, what: str) -> str:
    ref = part.get(key)
    url = ref if isinstance(ref, str) else obj(ref, f"{what}.{key}").get("url")
    if not isinstance(url, str):
        raise BadRequest(f"{what}: {key} needs a url")
    return url


def _user_content(content: object, what: str, images: list[bytes], videos: list[bytes]) -> str | list[JSONObject]:
    """A user message's content: text as `_text` reads it, or, with images or videos, the template's list of text,
    image and video items; each image's bytes go to `images`, each video's to `videos`."""
    if content is None or isinstance(content, str):
        return _text(content, what)
    parts = [obj(part, what) for part in array(content, what)]
    if all(p.get("type") == "text" for p in parts):
        return _text(content, what)
    items: list[JSONObject] = []
    for p in parts:
        match p.get("type"):
            case "text":
                items.append({"type": "text", "text": str(p.get("text", ""))})
            case "image_url":
                try:
                    images.append(from_data_url(_url(p, "image_url", what)))
                except ImageError as exc:
                    raise BadRequest(f"{what}: {exc}") from None
                items.append({"type": "image"})
            case "video_url":
                try:
                    videos.append(video.from_data_url(_url(p, "video_url", what)))
                except ImageError as exc:
                    raise BadRequest(f"{what}: {exc}") from None
                items.append({"type": "video"})
            case kind:
                raise BadRequest(f"{what}: only text, image_url and video_url parts are supported, not {kind!r}")
    return items


def settings_from(body: JSONObject, defaults: protocol.Settings) -> protocol.Settings:
    values: dict[str, int | float] = {}
    for key, name, kind in (
        ("temperature", "temperature", float),
        ("top_p", "top_p", float),
        ("top_k", "top_k", int),
        ("min_p", "min_p", float),
        ("presence_penalty", "penalty_present", float),
        ("frequency_penalty", "penalty_freq", float),
        ("repetition_penalty", "penalty_repeat", float),
        ("seed", "seed", int),
    ):
        value = number(body, key, kind)
        if value is not None:
            values[name] = value
    if "seed" in values:
        values["seed"] = int(values["seed"]) % 2**64
    return settings(defaults, values)


def _messages(raw: list[Any], images: list[bytes], videos: list[bytes]) -> list[JSONObject]:
    out: list[JSONObject] = []
    for i, m in enumerate(raw):
        m = obj(m, f"messages[{i}]")
        role = m.get("role")
        what = f"messages[{i}]"
        match role:
            case "system" | "developer":
                out.append({"role": "system", "content": _text(m.get("content"), what)})
            case "user":
                out.append({"role": "user", "content": _user_content(m.get("content"), what, images, videos)})
            case "assistant":
                msg: JSONObject = {"role": "assistant", "content": _text(m.get("content"), what)}
                reasoning = m.get("reasoning_content", m.get("reasoning"))
                if isinstance(reasoning, str):
                    msg["reasoning_content"] = reasoning
                calls: list[JSONObject] = []
                for j, c in enumerate(array(m.get("tool_calls") or [], f"{what}.tool_calls")):
                    fn = obj(obj(c, f"{what}.tool_calls[{j}]").get("function"), f"{what}.tool_calls[{j}].function")
                    arguments: object = fn.get("arguments", {})
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments) if arguments.strip() else {}
                        except json.JSONDecodeError:
                            raise BadRequest(f"{what}.tool_calls[{j}]: arguments are not JSON") from None
                    calls.append({"function": {"name": str(fn.get("name")), "arguments": obj(arguments, "arguments")}})
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
            case "tool":
                out.append({"role": "tool", "content": _text(m.get("content"), what)})
            case _:
                raise BadRequest(f"{what}: unknown role {role!r}")
    return out


def _tool_schemas(tools: list[JSONObject]) -> dict[str, Mapping[str, Any]]:
    schemas: dict[str, Mapping[str, Any]] = {}
    for t in tools:
        fn = obj(t.get("function"), "tools[].function")
        schemas[str(fn.get("name"))] = obj(fn.get("parameters") or {}, "parameters")
    return schemas


def parse(body: JSONObject, defaults: protocol.Settings) -> tuple[Request, bool]:
    """The request, and whether it streams."""
    images: list[bytes] = []
    videos: list[bytes] = []
    messages = _messages(array(body.get("messages"), "messages"), images, videos)
    if not messages:
        raise BadRequest("messages is empty")
    tools = [obj(t, "tools[]") for t in array(body.get("tools") or [], "tools")]
    if body.get("tool_choice") == "none":
        tools = []
    kwargs = obj(body.get("chat_template_kwargs") or {}, "chat_template_kwargs")
    effort = kwargs.get("reasoning_effort")
    if effort is None and isinstance(body.get("reasoning_effort"), str):
        effort = EFFORTS.get(body["reasoning_effort"])
        if effort is None:
            raise BadRequest(f"reasoning_effort {body['reasoning_effort']!r} is not one of {sorted(EFFORTS)}")
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and cast(JSONObject, fmt).get("type") in ("json_object", "json_schema"):
        messages = _with_json_instruction(messages, cast(JSONObject, fmt))
    conversation = Conversation(
        messages,
        tools or None,
        enable_thinking=boolean(kwargs.get("enable_thinking")),
        reasoning_effort=effort if isinstance(effort, str) else None,
        preserve_thinking=boolean(kwargs.get("preserve_thinking")),
    )
    stop = body.get("stop")
    stops = (stop,) if isinstance(stop, str) else tuple(str(s) for s in array(stop or [], "stop"))
    max_tokens = number(body, "max_completion_tokens", int) or number(body, "max_tokens", int)
    request = Request(
        conversation,
        settings_from(body, defaults),
        int(max_tokens) if max_tokens is not None else None,
        stops,
        _tool_schemas(tools) if tools else None,
        tuple(images),
        tuple(videos),
    )
    return request, bool(body.get("stream"))


def _with_json_instruction(messages: list[JSONObject], fmt: JSONObject) -> list[JSONObject]:
    """JSON output is prompted, then checked (as Strata's server does): the instruction goes in the system
    message."""
    instruction = JSON_INSTRUCTION
    schema = fmt.get("json_schema")
    if isinstance(schema, dict) and "schema" in schema:
        instruction += " It must follow this JSON schema: " + json.dumps(cast(JSONObject, schema)["schema"])
    if messages[0]["role"] == "system":
        first = {**messages[0], "content": f"{messages[0]['content']}\n\n{instruction}".strip()}
        return [first, *messages[1:]]
    return [{"role": "system", "content": instruction}, *messages]


def wants_json(body: JSONObject) -> bool:
    fmt = body.get("response_format")
    return isinstance(fmt, dict) and cast(JSONObject, fmt).get("type") in ("json_object", "json_schema")


def _call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def _call_json(call: ToolCall, index: int) -> JSONObject:
    return {
        "index": index,
        "id": _call_id(),
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
    }


def _usage(end: End) -> JSONObject:
    return {
        "prompt_tokens": end.prompt_tokens,
        "completion_tokens": end.completion_tokens,
        "total_tokens": end.prompt_tokens + end.completion_tokens,
        "prompt_tokens_details": {"cached_tokens": end.cached_tokens},
        **end.speeds(),
    }


def response(result: Result, model: str, check_json: bool) -> JSONObject:
    end = result.end
    assert end is not None
    content = "".join(result.content)
    if check_json and end.finish in (Finish.STOP, Finish.STOP_STRING):
        try:
            json.loads(content)
        except json.JSONDecodeError:
            raise BadRequest("the model's output is not valid JSON") from None
    message: JSONObject = {"role": "assistant", "content": content or None}
    if result.reasoning:
        message["reasoning_content"] = "".join(result.reasoning)
    if result.calls:
        message["tool_calls"] = [_call_json(c, i) for i, c in enumerate(result.calls)]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": _FINISH[end.finish]}],
        "usage": _usage(end),
    }


async def stream_events(
    pieces: AsyncIterator[Piece | End], model: str, include_usage: bool
) -> AsyncIterator[JSONObject | None]:
    """Chunks for server-sent events; None marks the end (`data: [DONE]`)."""
    rid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: JSONObject, finish: str | None = None) -> JSONObject:
        return {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    yield chunk({"role": "assistant", "content": ""})
    calls = 0
    async for piece in pieces:
        match piece:
            case Reasoning(text=t):
                yield chunk({"reasoning_content": t})
            case Content(text=t):
                yield chunk({"content": t})
            case ToolCallStart(name=name):
                call = {
                    "index": calls,
                    "id": _call_id(),
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }
                yield chunk({"tool_calls": [call]})
            case ToolCall(arguments=arguments):
                arguments = json.dumps(arguments, ensure_ascii=False)
                yield chunk({"tool_calls": [{"index": calls, "function": {"arguments": arguments}}]})
                calls += 1
            case End():
                if piece.error is not None:
                    yield {"error": {"message": piece.error, "type": "server_error"}}
                yield chunk({}, _FINISH[piece.finish])
                if include_usage:
                    yield {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [],
                        "usage": _usage(piece),
                    }
    yield None
