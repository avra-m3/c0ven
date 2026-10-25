"""One request through the pipeline both APIs share: render, tokenize, generate, detokenize, parse.

A prompt that fills the context is refused, never truncated; a `max_tokens` past the context's end is cut to the rest
of the context (DESIGN.md, section 2).
Stop strings are matched in the content as it streams; the first one ends the output and stops the request.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import protocol
from frontend import video
from frontend.images import IMAGE_PAD, ImageError, PromptImage, expand, prepare
from frontend.model import Model
from frontend.output import Content, OutputParser, Piece, Reasoning, ToolCall, ToolCallError, ToolCallStart
from frontend.template import Conversation, TemplateFailed
from frontend.tokenizer import Detokenizer, encode


class BadRequest(ValueError):
    """The client's request is wrong: answered with 400."""


class Engine(Protocol):
    @property
    def max_context(self) -> int: ...

    @property
    def image_tokens_max(self) -> int: ...

    def generate(
        self,
        prompt: Sequence[int],
        max_new: int,
        settings: protocol.Settings,
        images: Sequence[PromptImage] = (),
    ) -> AsyncGenerator[protocol.Event]: ...


class Finish(StrEnum):
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    STOP_STRING = "stop_string"
    CANCEL = "cancel"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class Request:
    conversation: Conversation
    settings: protocol.Settings
    max_tokens: int | None = None
    stop: tuple[str, ...] = ()
    tool_schemas: Mapping[str, Mapping[str, Any]] | None = None
    images: tuple[bytes, ...] = ()
    """The images' encoded bytes, in the order the conversation's image parts come."""
    videos: tuple[bytes, ...] = ()
    """The videos' encoded bytes, in the order the conversation's video parts come."""


@dataclass(slots=True)
class Prompt:
    text: str
    ids: list[int]
    thinking: bool
    """The rendered prompt ends inside `<think>`: the output starts as reasoning."""
    images: list[PromptImage] = field(default_factory=lambda: [])


@dataclass(frozen=True, slots=True)
class End:
    finish: Finish
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    stop_string: str | None = None
    error: str | None = None
    prompt_ms: float = 0.0
    """From the request's admission to its first token, as the engine timed it; 0 when it sent no time."""
    decode_ms: float = 0.0
    """From the first token to the last."""
    queue_ms: float = 0.0
    """From the request's GEN to its admission."""
    steps: int = 0
    """The DONE line's counters: the decode steps and rounds that gave its tokens, the drafts verified and accepted
    for it, and of those the ones from its own tokens."""
    drafts: int = 0
    accepted: int = 0
    lookup_drafts: int = 0
    lookup_accepted: int = 0

    def speeds(self) -> dict[str, float]:
        """Prompt and decode speeds in tokens a second, and the time in seconds, under TabbyAPI's `usage` names,
        which proxies read; empty without the engine's times."""
        if self.prompt_ms <= 0:
            return {}
        out = {
            "prompt_tokens_per_sec": round((self.prompt_tokens - self.cached_tokens) * 1e3 / self.prompt_ms, 2),
            "total_time": round((self.prompt_ms + self.decode_ms) / 1e3, 3),
        }
        if self.decode_ms > 0 and self.completion_tokens > 1:
            out["completion_tokens_per_sec"] = round((self.completion_tokens - 1) * 1e3 / self.decode_ms, 2)
        return out


@dataclass(slots=True)
class Result:
    reasoning: list[str] = field(default_factory=lambda: [])
    content: list[str] = field(default_factory=lambda: [])
    calls: list[ToolCall] = field(default_factory=lambda: [])
    end: End | None = None


def _pad(model: Model, token: str) -> int:
    pad = model.tokenizer.token_to_id(token)
    if pad is None:
        raise BadRequest(f"the model has no {token} token")
    return pad


def render(model: Model, request: Request, image_tokens_max: int = 0) -> Prompt:
    """The prompt's text and ids; with images and videos, each image and each pair of a video's frames sized to at
    most `image_tokens_max` tokens. A video's template item becomes its frame pairs' text before tokenizing."""
    if (request.images or request.videos) and not image_tokens_max:
        raise BadRequest("this engine reads no images or videos")
    try:
        prepared = [prepare(raw, image_tokens_max) for raw in request.images]
        videos = [video.prepare(raw, image_tokens_max) for raw in request.videos]
    except ImageError as exc:
        raise BadRequest(str(exc)) from None
    try:
        text = model.template.render(request.conversation)
    except TemplateFailed as exc:
        raise BadRequest(f"the chat template refused the conversation: {exc}") from None
    if videos:
        parts = text.split(video.TEMPLATE_ITEM)
        if len(parts) != len(videos) + 1:
            raise BadRequest(f"the rendered prompt has {len(parts) - 1} video places for {len(videos)} videos")
        text = parts[0] + "".join(video.placeholder(v) + rest for v, rest in zip(videos, parts[1:], strict=True))
    ids = encode(model.tokenizer, text)
    images: list[PromptImage] = []
    if prepared or videos:
        pairs = [(p.grid_h, p.grid_w, p.pixels) for v in videos for p in v]
        try:
            ids, images = expand(
                ids,
                _pad(model, IMAGE_PAD) if prepared else -1,
                prepared,
                _pad(model, video.VIDEO_PAD) if videos else -1,
                pairs,
            )
        except ImageError as exc:
            raise BadRequest(str(exc)) from None
    return Prompt(text, ids, text.endswith("<think>\n"), images)


def budget(engine: Engine, prompt: Prompt, max_tokens: int | None, default: int = 0) -> int:
    """The tokens a request may write: max_tokens, cut to the rest of the context. default: the limit for a request
    without max_tokens, 0 for the rest of the context. The engine reserves KV for all of them when it admits the
    request. Agent clients send one max_tokens however long the conversation grows, so a limit past the context's
    end is cut. Raises BadRequest when max_tokens is below 1 or the prompt fills the context."""
    if max_tokens is not None and max_tokens < 1:
        raise BadRequest("max_tokens must be at least 1")
    room = engine.max_context - len(prompt.ids)
    if room < 1:
        raise BadRequest(f"the prompt ({len(prompt.ids)} tokens) fills the context ({engine.max_context})")
    if max_tokens is None:
        return min(room, default) if default else room
    return min(room, max_tokens)


class _StopStrings:
    """Holds back content that could be the start of a stop string."""

    def __init__(self, stops: Sequence[str]) -> None:
        self.stops = [s for s in stops if s]
        self.pending = ""

    def push(self, text: str) -> tuple[str, str | None]:
        if not self.stops:
            return text, None
        self.pending += text
        hits = [(i, s) for s in self.stops if (i := self.pending.find(s)) >= 0]
        if hits:
            i, s = min(hits)
            out, self.pending = self.pending[:i], ""
            return out, s
        hold = 0
        for s in self.stops:
            for n in range(min(len(s) - 1, len(self.pending)), 0, -1):
                if s.startswith(self.pending[-n:]):
                    hold = max(hold, n)
                    break
        out = self.pending[: len(self.pending) - hold]
        self.pending = self.pending[len(out) :]
        return out, None

    def flush(self) -> str:
        out, self.pending = self.pending, ""
        return out


async def stream(model: Model, engine: Engine, request: Request, prompt: Prompt) -> AsyncIterator[Piece | End]:
    """The output as pieces, then one End."""
    max_new = budget(engine, prompt, request.max_tokens)
    detok = Detokenizer(model.tokenizer)
    parser = OutputParser(prompt.thinking, request.tool_schemas, tools=request.tool_schemas is not None)
    stops = _StopStrings(request.stop)
    generated = 0
    cached = 0

    def pieces(text: str) -> list[Piece]:
        return list(parser.feed(text))

    async with aclosing(engine.generate(prompt.ids, max_new, request.settings, prompt.images)) as events:
        async for event in events:
            match event:
                case protocol.Resume(reused=reused):
                    cached = reused
                case protocol.Tokens(ids=ids):
                    generated += len(ids)
                    text = detok.push([i for i in ids if i not in model.stop_ids])
                    for piece in pieces(text):
                        if isinstance(piece, Content):
                            out, hit = stops.push(piece.text)
                            if out:
                                yield Content(out)
                            if hit is not None:
                                yield End(Finish.STOP_STRING, len(prompt.ids), generated, cached, stop_string=hit)
                                return
                        else:
                            yield piece
                case protocol.Done(finish=finish, reused=reused, prompt_ms=prompt_ms, decode_ms=decode_ms) as done:
                    cached = reused or cached
                    try:
                        tail = pieces(detok.flush()) + list(parser.finish())
                    except ToolCallError as exc:
                        yield End(Finish.ERROR, len(prompt.ids), generated, cached, error=str(exc))
                        return
                    for piece in tail:
                        if isinstance(piece, Content):
                            out, hit = stops.push(piece.text)
                            if out:
                                yield Content(out)
                            if hit is not None:
                                yield End(Finish.STOP_STRING, len(prompt.ids), generated, cached, stop_string=hit)
                                return
                        else:
                            yield piece
                    if rest := stops.flush():
                        yield Content(rest)
                    why = {
                        protocol.Finish.STOP: Finish.STOP,
                        protocol.Finish.LENGTH: Finish.LENGTH,
                        protocol.Finish.CANCEL: Finish.CANCEL,
                    }[finish]
                    if why is Finish.STOP and parser.calls:
                        why = Finish.TOOL_CALLS
                    yield End(
                        why,
                        len(prompt.ids),
                        generated,
                        cached,
                        prompt_ms=prompt_ms,
                        decode_ms=decode_ms,
                        queue_ms=done.queue_ms,
                        steps=done.steps,
                        drafts=done.drafts,
                        accepted=done.accepted,
                        lookup_drafts=done.lookup_drafts,
                        lookup_accepted=done.lookup_accepted,
                    )
                    return
                case protocol.Error(message=message):
                    yield End(Finish.ERROR, len(prompt.ids), generated, cached, error=message)
                    return
                case _:
                    pass


async def collect(pieces: AsyncIterator[Piece | End]) -> Result:
    result = Result()
    async for piece in pieces:
        match piece:
            case Reasoning(text=t):
                result.reasoning.append(t)
            case Content(text=t):
                result.content.append(t)
            case ToolCallStart():
                pass
            case ToolCall():
                result.calls.append(piece)
            case End():
                result.end = piece
    return result
