"""The admin routes (ADR-0032): wake, sleep, and the lease that llama-swap's command holds.

They answer on the admin port only, which the front end binds apart from the API's, so a client of the API cannot
put the engine to sleep.

| Endpoint | What |
| --- | --- |
| `POST /admin/wake` | loads the engine; answers with its INFO once it is READY, or 503 with why not |
| `POST /admin/sleep[?drain_ms=n]` | drains, gives the cards back, ends every lease; answers with the engine's ms |
| `POST /admin/lease` | wakes the engine, then holds the response open, one line every few seconds |

A lease fails closed: once the last lease's connection ends, for whatever reason, the front end sleeps on its own,
so the engine cannot stay on the cards after llama-swap believes it gone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from frontend.engine_client import EngineUnavailable

log = logging.getLogger("frontend.admin")

type Handler = Callable[[Request], Coroutine[Any, Any, Response]]


class SleepingEngine(Protocol):
    @property
    def asleep(self) -> bool: ...

    async def wake(self) -> dict[str, str]: ...

    async def sleep(self, drain_ms: int) -> float: ...


@dataclass(frozen=True, slots=True)
class Admin:
    drain_ms: int = 30_000
    """How long a sleep lets the running requests go on before it cancels them."""
    port: int | None = None
    """The admin port: the routes answer there only. None: on every port, for the tests."""
    heartbeat_s: float = 10.0


class _Held(StreamingResponse):
    """A lease's response, which lets go of its lease however it ends, even before its first line."""

    def __init__(self, content: AsyncIterator[str], release: Callable[[], None]) -> None:
        super().__init__(content, media_type="application/x-ndjson")
        self._release = release

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._release()


class Leases:
    """The leases held, and the one order of wakes and sleeps."""

    def __init__(self, engine: SleepingEngine, admin: Admin) -> None:
        self._engine = engine
        self._admin = admin
        self._held: set[asyncio.Event] = set()
        """One per lease; set when a sleep ends it."""
        self._lock = asyncio.Lock()
        self._lapses: set[asyncio.Task[None]] = set()

    @property
    def count(self) -> int:
        return len(self._held)

    async def wake(self) -> dict[str, str]:
        async with self._lock:
            return await self._engine.wake()

    async def sleep(self, drain_ms: int) -> float:
        async with self._lock:
            ms = await self._engine.sleep(drain_ms)
        for ended in self._held:
            ended.set()
        return ms

    def take(self) -> asyncio.Event:
        """A new lease, counted before its wake, so a lapse meanwhile does not sleep the engine under it."""
        ended = asyncio.Event()
        self._held.add(ended)
        return ended

    def release(self, ended: asyncio.Event) -> None:
        self._held.discard(ended)
        if self._held or ended.is_set():
            return
        task = asyncio.get_running_loop().create_task(self._lapse(), name="lease-lapse")
        self._lapses.add(task)
        task.add_done_callback(self._lapses.discard)

    async def _lapse(self) -> None:
        async with self._lock:
            if self._held or self._engine.asleep:
                return
            log.warning("the last lease is gone: the engine sleeps")
            await self._engine.sleep(self._admin.drain_ms)

    async def lines(self, ended: asyncio.Event, info: dict[str, str]) -> AsyncIterator[str]:
        yield json.dumps({"status": "ready", "engine": info}) + "\n"
        while not ended.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(ended.wait(), self._admin.heartbeat_s)
            if not ended.is_set():
                yield "\n"
        yield json.dumps({"status": "asleep"}) + "\n"


def routes(engine: SleepingEngine, admin: Admin) -> tuple[list[Route], Leases]:
    leases = Leases(engine, admin)

    def admin_only(handler: Handler) -> Handler:
        async def wrapped(request: Request) -> Response:
            server = request.scope.get("server") or (None, None)
            if admin.port is not None and server[1] != admin.port:
                return JSONResponse({"error": "not found"}, status_code=404)
            try:
                return await handler(request)
            except EngineUnavailable as exc:
                return JSONResponse({"status": "asleep", "error": str(exc)}, status_code=503)

        return wrapped

    async def wake(_: Request) -> Response:
        return JSONResponse({"status": "ready", "engine": await leases.wake()})

    async def sleep(request: Request) -> Response:
        drain_ms = admin.drain_ms
        if (value := request.query_params.get("drain_ms")) is not None:
            if not value.isdigit():
                return JSONResponse({"error": "drain_ms must be a whole number of milliseconds"}, status_code=400)
            drain_ms = int(value)
        ms = await leases.sleep(drain_ms)
        return JSONResponse({"status": "asleep", "ms": round(ms, 1)})

    async def lease(_: Request) -> Response:
        ended = leases.take()
        try:
            info = await leases.wake()
        except BaseException:
            leases.release(ended)
            raise
        return _Held(leases.lines(ended, info), lambda: leases.release(ended))

    return [
        Route("/admin/wake", admin_only(wake), methods=["POST"]),
        Route("/admin/sleep", admin_only(sleep), methods=["POST"]),
        Route("/admin/lease", admin_only(lease), methods=["POST"]),
    ], leases
