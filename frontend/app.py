"""The HTTP server (ADR-0004: Starlette on uvicorn). Endpoints follow Strata's server
(spec/STRATA_DECODE.md, section 10), with several requests in flight at once (ADR-0010, item 6).

| Endpoint | What |
| --- | --- |
| `POST /v1/chat/completions` | OpenAI Chat Completions |
| `POST /v1/messages`, `/v1/messages/count_tokens` | Anthropic Messages |
| `GET /v1/models`, `/models` | the loaded model |
| `GET /health`, `/api/health` | the engine's status and context: 200 once it is READY, 503 while it loads or sleeps |
| `GET /props` | chat template, context, generation defaults, model path |
| `GET /slots` | the engine's slots and how many are busy |
| `GET /metrics` | the engine's facts (its INFO line) and the front end's counters, as JSON |

With an `Admin`, the routes of `frontend/admin.py` too. While the engine sleeps, every request is refused with 503
before it reaches the engine, so none wakes it (ADR-0032).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from frontend import admin as admin_routes
from frontend import anthropic_api, generate, openai_api
from frontend.admin import Admin
from frontend.body import obj
from frontend.engine_client import EngineUnavailable
from frontend.generate import BadRequest, Engine
from frontend.model import Model
from frontend.output import Piece

log = logging.getLogger("frontend")

type JSONObject = dict[str, Any]
type Handler = Callable[[Request], Coroutine[Any, Any, Response]]


class ServedEngine(Engine, Protocol):
    """The engine as the app needs it: the generation interface, plus its facts for the status endpoints."""

    @property
    def info(self) -> dict[str, str]: ...

    @property
    def slots(self) -> int: ...

    @property
    def busy(self) -> int: ...

    @property
    def status(self) -> str: ...

    @property
    def asleep(self) -> bool: ...

    async def start(self) -> None: ...

    async def wake(self) -> dict[str, str]: ...

    async def sleep(self, drain_ms: int) -> float: ...

    async def close(self) -> None: ...


@dataclass(slots=True)
class Counters:
    requests: int = 0
    errors: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    recent: list[JSONObject] = field(default_factory=lambda: [])

    def record(self, end: generate.End, api: str, seconds: float) -> None:
        self.requests += 1
        self.prompt_tokens += end.prompt_tokens
        self.completion_tokens += end.completion_tokens
        self.cached_tokens += end.cached_tokens
        if end.error is not None:
            self.errors += 1
        self.recent.append(
            {
                "api": api,
                "finish": str(end.finish),
                "prompt_tokens": end.prompt_tokens,
                "completion_tokens": end.completion_tokens,
                "cached_tokens": end.cached_tokens,
                "seconds": round(seconds, 3),
            }
        )
        del self.recent[:-12]


def error(status: int, message: str, kind: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse({"type": "error", "error": {"type": kind, "message": message}}, status_code=status)


async def _body(request: Request) -> JSONObject:
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise BadRequest("the body is not JSON") from None
    if not isinstance(body, dict):
        raise BadRequest("the body must be a JSON object")
    return body  # pyright: ignore[reportUnknownVariableType]


def _sse(data: JSONObject | None, event: str | None = None) -> bytes:
    payload = "[DONE]" if data is None else json.dumps(data, ensure_ascii=False)
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


def create_app(
    model: Model, engine: ServedEngine, admin: Admin | None = None, default_max_tokens: int = 0
) -> Starlette:
    """default_max_tokens: what a request without max_tokens may write; 0 for the rest of the context."""
    counters = Counters()

    def admitting() -> None:
        if engine.asleep:
            raise EngineUnavailable(f"the engine is {engine.status}")

    def guarded(handler: Handler) -> Handler:
        async def wrapped(request: Request) -> Response:
            try:
                return await handler(request)
            except BadRequest as exc:
                return error(400, str(exc))
            except EngineUnavailable as exc:
                return error(503, str(exc), "overloaded_error")

        return wrapped

    async def recorded(pieces: AsyncIterator[Piece | generate.End], api: str) -> AsyncIterator[Piece | generate.End]:
        t0 = time.monotonic()
        async for piece in pieces:
            if isinstance(piece, generate.End):
                counters.record(piece, api, time.monotonic() - t0)
            yield piece

    async def render(req: generate.Request) -> generate.Prompt:
        """The prompt; images and videos are decoded and resized on a thread of their own, to the running engine's
        limit."""
        if not req.images and not req.videos:
            return generate.render(model, req)
        await engine.start()
        return await asyncio.to_thread(generate.render, model, req, engine.image_tokens_max)

    async def chat_completions(request: Request) -> Response:
        admitting()
        body = await _body(request)
        req, streaming = openai_api.parse(body, model.defaults)
        prompt = await render(req)
        req = replace(req, max_tokens=generate.budget(engine, prompt, req.max_tokens, default_max_tokens))
        pieces = recorded(generate.stream(model, engine, req, prompt), "openai")
        if not streaming:
            result = await generate.collect(pieces)
            assert result.end is not None
            if result.end.error is not None:
                return error(500, result.end.error, "server_error")
            return JSONResponse(openai_api.response(result, model.name, openai_api.wants_json(body)))
        usage = bool(obj(body.get("stream_options") or {}, "stream_options").get("include_usage"))

        async def events() -> AsyncIterator[bytes]:
            async for chunk in openai_api.stream_events(pieces, model.name, usage):
                yield _sse(chunk)

        return StreamingResponse(events(), media_type="text/event-stream")

    async def messages(request: Request) -> Response:
        admitting()
        body = await _body(request)
        req, streaming = anthropic_api.parse(body, model.defaults)
        prompt = await render(req)
        req = replace(req, max_tokens=generate.budget(engine, prompt, req.max_tokens, default_max_tokens))
        pieces = recorded(generate.stream(model, engine, req, prompt), "anthropic")
        if not streaming:
            result = await generate.collect(pieces)
            assert result.end is not None
            if result.end.error is not None:
                return error(500, result.end.error, "api_error")
            return JSONResponse(anthropic_api.response(result, model.name))

        async def events() -> AsyncIterator[bytes]:
            async for name, data in anthropic_api.stream_events(pieces, model.name, len(prompt.ids)):
                yield _sse(data, name)

        return StreamingResponse(events(), media_type="text/event-stream")

    async def count_tokens(request: Request) -> Response:
        body = await _body(request)
        images: list[bytes] = []
        videos: list[bytes] = []
        conv, _ = anthropic_api.conversation(body, images, videos)
        req = generate.Request(conv, model.defaults, images=tuple(images), videos=tuple(videos))
        return JSONResponse({"input_tokens": len((await render(req)).ids)})

    async def models(_: Request) -> Response:
        return JSONResponse(
            {"object": "list", "data": [{"id": model.name, "object": "model", "owned_by": "local", "created": 0}]}
        )

    async def health(_: Request) -> Response:
        status = engine.status
        ready = status == "ready"
        return JSONResponse(
            {
                "status": "ok" if ready else status,
                "context": engine.max_context,
                "images": engine.image_tokens_max > 0,
                "loaded": ready,
            },
            status_code=200 if ready else 503,
        )

    async def props(_: Request) -> Response:
        d = model.defaults
        return JSONResponse(
            {
                "chat_template": model.template.source,
                "context": engine.max_context,
                "model_path": model.path,
                "generation_settings": {
                    "temperature": d.temperature,
                    "top_k": d.top_k,
                    "top_p": d.top_p,
                    "max_tokens": default_max_tokens or None,
                },
                "engine": engine.info,
            }
        )

    async def slots(_: Request) -> Response:
        return JSONResponse({"slots": engine.slots, "busy": engine.busy, "n_ctx": engine.max_context})

    async def metrics(_: Request) -> Response:
        c = counters
        return JSONResponse(
            {
                "engine": engine.info,
                "busy": engine.busy,
                "totals": {
                    "requests": c.requests,
                    "errors": c.errors,
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "cached_tokens": c.cached_tokens,
                },
                "recent": c.recent,
            }
        )

    @asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None]:
        await engine.start()
        yield
        await engine.close()

    routes = [
        Route("/v1/chat/completions", guarded(chat_completions), methods=["POST"]),
        Route("/v1/messages", guarded(messages), methods=["POST"]),
        Route("/v1/messages/count_tokens", guarded(count_tokens), methods=["POST"]),
        Route("/v1/models", models),
        Route("/models", models),
        Route("/health", health),
        Route("/api/health", health),
        Route("/props", props),
        Route("/slots", slots),
        Route("/metrics", metrics),
    ]
    if admin is not None:
        routes += admin_routes.routes(engine, admin)[0]
    return Starlette(routes=routes, lifespan=lifespan)
