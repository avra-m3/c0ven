"""Anthropic Messages: `POST /v1/messages` and `/v1/messages/count_tokens`, streamed and not, with tools and
thinking.

Content blocks map onto the template's messages: `thinking` becomes the assistant's `reasoning_content`,
`tool_use` its tool calls, and each `tool_result` a `tool` message. `thinking: {"type": "disabled"}` turns the
template's thinking off; without it, the template's default (on) holds.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any, cast

import protocol
from frontend.body import array, boolean, obj, settings
from frontend.generate import BadRequest, End, Finish, Request, Result
from frontend.images import ImageError, from_base64
from frontend.openai_api import EFFORTS
from frontend.output import Content, Piece, Reasoning, ToolCall
from frontend.template import Conversation

type JSONObject = dict[str, Any]

_STOP_REASON = {
    Finish.STOP: "end_turn",
    Finish.STOP_STRING: "stop_sequence",
    Finish.LENGTH: "max_tokens",
    Finish.TOOL_CALLS: "tool_use",
    Finish.CANCEL: "end_turn",
    Finish.ERROR: "end_turn",
}


def _blocks(content: object, what: str) -> list[JSONObject]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [obj(b, what) for b in array(content, what)]


def _block_text(content: object, what: str) -> str:
    parts: list[str] = []
    for b in _blocks(content, what):
        if b.get("type") != "text":
            raise BadRequest(f"{what}: only text blocks are supported here, not {b.get('type')!r}")
        parts.append(str(b.get("text", "")))
    return "".join(parts)


def _media(block: JSONObject, what: str, kind: str = "image") -> bytes:
    """An image or video block's bytes: a base64 source of that kind. URL sources are refused: this engine does not
    fetch. A `video` block is this engine's own, shaped as an image block."""
    source = obj(block.get("source"), f"{what}.source")
    if source.get("type") != "base64" or not str(source.get("media_type", "")).startswith(f"{kind}/"):
        raise BadRequest(f"{what}: a {kind}'s source must be base64 {kind} data")
    try:
        return from_base64(str(source.get("data", "")))
    except ImageError as exc:
        raise BadRequest(f"{what}: {exc}") from None


def _messages(system: object, raw: list[Any], images: list[bytes], videos: list[bytes]) -> list[JSONObject]:
    out: list[JSONObject] = []
    if system:
        out.append({"role": "system", "content": _block_text(system, "system")})
    for i, m in enumerate(raw):
        m = obj(m, f"messages[{i}]")
        what = f"messages[{i}]"
        role = m.get("role")
        blocks = _blocks(m.get("content"), what)
        if role == "user":
            text: list[str] = []
            items: list[JSONObject] = []
            for b in blocks:
                match b.get("type"):
                    case "text":
                        text.append(str(b.get("text", "")))
                        items.append({"type": "text", "text": text[-1]})
                    case "image":
                        images.append(_media(b, what))
                        items.append({"type": "image"})
                    case "video":
                        videos.append(_media(b, what, "video"))
                        items.append({"type": "video"})
                    case "tool_result":
                        result = b.get("content", "")
                        body = _block_text(result, f"{what}.tool_result") if result else ""
                        if b.get("is_error"):
                            body = f"Error: {body}"
                        out.append({"role": "tool", "content": body})
                    case kind:
                        raise BadRequest(f"{what}: unsupported block {kind!r}")
            if len(items) > len(text):
                out.append({"role": "user", "content": items})
            elif text:
                out.append({"role": "user", "content": "".join(text)})
        elif role == "assistant":
            msg: JSONObject = {"role": "assistant", "content": ""}
            calls: list[JSONObject] = []
            for b in blocks:
                match b.get("type"):
                    case "text":
                        msg["content"] += str(b.get("text", ""))
                    case "thinking":
                        msg["reasoning_content"] = msg.get("reasoning_content", "") + str(b.get("thinking", ""))
                    case "redacted_thinking":
                        pass
                    case "tool_use":
                        calls.append(
                            {"function": {"name": str(b.get("name")), "arguments": obj(b.get("input", {}), "input")}}
                        )
                    case kind:
                        raise BadRequest(f"{what}: unsupported block {kind!r}")
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            raise BadRequest(f"{what}: unknown role {role!r}")
    return out


def settings_from(body: JSONObject, defaults: protocol.Settings) -> protocol.Settings:
    values: dict[str, int | float] = {}
    for key, kind in (("temperature", float), ("top_p", float), ("top_k", int)):
        value = body.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BadRequest(f"{key} must be a number")
        values[key] = kind(value)
    return settings(defaults, values)


def conversation(
    body: JSONObject, images: list[bytes] | None = None, videos: list[bytes] | None = None
) -> tuple[Conversation, dict[str, Any] | None]:
    """The conversation and the tools' schemas; the images' bytes go to `images`, the videos' to `videos`."""
    messages = _messages(
        body.get("system"),
        array(body.get("messages"), "messages"),
        [] if images is None else images,
        [] if videos is None else videos,
    )
    if not messages:
        raise BadRequest("messages is empty")
    tools: list[JSONObject] = []
    schemas: dict[str, Any] = {}
    for t in array(body.get("tools") or [], "tools"):
        t = obj(t, "tools[]")
        schema = obj(t.get("input_schema") or {}, "input_schema")
        tools.append(
            {
                "type": "function",
                "function": {"name": t.get("name"), "description": t.get("description", ""), "parameters": schema},
            }
        )
        schemas[str(t.get("name"))] = schema
    choice = body.get("tool_choice")
    if isinstance(choice, dict) and cast(JSONObject, choice).get("type") == "none":
        tools, schemas = [], {}
    thinking = body.get("thinking")
    enable: bool | None = None
    if isinstance(thinking, dict):
        kind = cast(JSONObject, thinking).get("type")
        enable = False if kind == "disabled" else True if kind in ("enabled", "adaptive") else None
    effort = body.get("reasoning_effort")
    if effort is not None and EFFORTS.get(str(effort)) is None:
        raise BadRequest(f"reasoning_effort {effort!r} is not one of {sorted(EFFORTS)}")
    kwargs = obj(body.get("chat_template_kwargs") or {}, "chat_template_kwargs")
    return (
        Conversation(
            messages,
            tools or None,
            enable_thinking=boolean(kwargs.get("enable_thinking", enable)),
            reasoning_effort=kwargs.get("reasoning_effort", EFFORTS.get(str(effort)) if effort else None),
            preserve_thinking=boolean(kwargs.get("preserve_thinking")),
        ),
        schemas or None,
    )


def parse(body: JSONObject, defaults: protocol.Settings) -> tuple[Request, bool]:
    images: list[bytes] = []
    videos: list[bytes] = []
    conv, schemas = conversation(body, images, videos)
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool):
        raise BadRequest("max_tokens is required, and must be an integer")
    stops = tuple(str(s) for s in array(body.get("stop_sequences") or [], "stop_sequences"))
    request = Request(conv, settings_from(body, defaults), max_tokens, stops, schemas, tuple(images), tuple(videos))
    return request, bool(body.get("stream"))


def _content(result: Result) -> list[JSONObject]:
    blocks: list[JSONObject] = []
    if result.reasoning:
        blocks.append({"type": "thinking", "thinking": "".join(result.reasoning), "signature": ""})
    if result.content:
        blocks.append({"type": "text", "text": "".join(result.content)})
    for call in result.calls:
        blocks.append(
            {"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": call.name, "input": call.arguments}
        )
    return blocks


def _usage(end: End) -> JSONObject:
    return {
        "input_tokens": end.prompt_tokens - end.cached_tokens,
        "cache_read_input_tokens": end.cached_tokens,
        "cache_creation_input_tokens": 0,
        "output_tokens": end.completion_tokens,
        **end.speeds(),
    }


def response(result: Result, model: str) -> JSONObject:
    end = result.end
    assert end is not None
    return {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": _content(result),
        "stop_reason": _STOP_REASON[end.finish],
        "stop_sequence": end.stop_string,
        "usage": _usage(end),
    }


type SSE = tuple[str, JSONObject]


class _Blocks:
    """The content blocks of a streamed message: one open at a time, numbered in order."""

    def __init__(self) -> None:
        self.index = -1
        self.open: str | None = None

    def start(self, kind: str, block: JSONObject) -> list[SSE]:
        out = self.close()
        self.index += 1
        self.open = kind
        return [
            *out,
            ("content_block_start", {"type": "content_block_start", "index": self.index, "content_block": block}),
        ]

    def close(self) -> list[SSE]:
        if self.open is None:
            return []
        out: list[SSE] = []
        if self.open == "thinking":
            out.append(self.delta({"type": "signature_delta", "signature": ""}))
        self.open = None
        out.append(("content_block_stop", {"type": "content_block_stop", "index": self.index}))
        return out

    def delta(self, delta: JSONObject) -> SSE:
        return "content_block_delta", {"type": "content_block_delta", "index": self.index, "delta": delta}


async def stream_events(pieces: AsyncIterator[Piece | End], model: str, prompt_tokens: int) -> AsyncIterator[SSE]:
    """(event name, data) pairs for server-sent events."""
    message: JSONObject = {
        "id": f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": prompt_tokens, "output_tokens": 0},
    }
    yield "message_start", {"type": "message_start", "message": message}
    blocks = _Blocks()
    async for piece in pieces:
        out: list[SSE] = []
        match piece:
            case Reasoning(text=t):
                if blocks.open != "thinking":
                    out += blocks.start("thinking", {"type": "thinking", "thinking": ""})
                out.append(blocks.delta({"type": "thinking_delta", "thinking": t}))
            case Content(text=t):
                if blocks.open != "text":
                    out += blocks.start("text", {"type": "text", "text": ""})
                out.append(blocks.delta({"type": "text_delta", "text": t}))
            case ToolCall():
                block: JSONObject = {
                    "type": "tool_use",
                    "id": f"toolu_{uuid.uuid4().hex[:24]}",
                    "name": piece.name,
                    "input": {},
                }
                out += blocks.start("tool_use", block)
                arguments = json.dumps(piece.arguments, ensure_ascii=False)
                out.append(blocks.delta({"type": "input_json_delta", "partial_json": arguments}))
                out += blocks.close()
            case End():
                out += blocks.close()
                if piece.error is not None:
                    out.append(("error", {"type": "error", "error": {"type": "api_error", "message": piece.error}}))
                else:
                    stop = {"stop_reason": _STOP_REASON[piece.finish], "stop_sequence": piece.stop_string}
                    usage = {"output_tokens": piece.completion_tokens, **piece.speeds()}
                    out.append(("message_delta", {"type": "message_delta", "delta": stop, "usage": usage}))
                    out.append(("message_stop", {"type": "message_stop"}))
        for event in out:
            yield event
