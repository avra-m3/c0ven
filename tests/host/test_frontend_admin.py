"""Sleep and wake from the front end (ADR-0032): the guards, the admin routes and the lease, with a
fake engine behind them."""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Awaitable, Callable

import uvicorn
from starlette.applications import Starlette
from starlette.testclient import TestClient
from tokenizers import Tokenizer

from frontend import lease
from frontend.admin import Admin
from frontend.app import create_app
from frontend.model import Model
from tests.helpers import FakeEngine

CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def test_asleep_health_is_503_and_no_request_reaches_the_engine(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer, status="asleep")
    with TestClient(create_app(model, engine, Admin())) as client:
        health = client.get("/health")
        assert (health.status_code, health.json()["status"]) == (503, "asleep")
        for path, body in (("/v1/chat/completions", CHAT), ("/v1/messages", {**CHAT, "max_tokens": 8})):
            r = client.post(path, json={**body, "stream": True})
            assert r.status_code == 503, path
            assert "asleep" in r.json()["error"]["message"]
        assert engine.prompts == []
        r = client.post("/admin/wake")
        assert (r.status_code, r.json()["status"]) == (200, "ready")
        assert client.get("/health").status_code == 200
        assert client.post("/v1/chat/completions", json=CHAT).status_code == 200


def test_sleep_takes_the_drain_from_the_query_or_the_default(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer)
    with TestClient(create_app(model, engine, Admin(drain_ms=5000))) as client:
        assert client.post("/admin/sleep?drain_ms=250").json() == {"status": "asleep", "ms": 1.5}
        assert client.post("/admin/sleep").status_code == 200
        assert client.post("/admin/sleep?drain_ms=soon").status_code == 400
        assert engine.sleeps == [250, 5000]


def test_a_failed_wake_answers_503_and_stays_asleep(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer, status="asleep", wake_error="the cards are taken")
    with TestClient(create_app(model, engine, Admin())) as client:
        r = client.post("/admin/wake")
        assert r.status_code == 503
        assert "the cards are taken" in r.json()["error"]
        assert client.post("/admin/lease").status_code == 503
        assert client.get("/health").status_code == 503


def test_the_admin_routes_answer_on_the_admin_port_only(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer)
    app = create_app(model, engine, Admin(port=8081))
    with TestClient(app, base_url="http://testserver:8080") as client:
        assert client.post("/admin/sleep").status_code == 404
    with TestClient(app, base_url="http://testserver:8081") as client:
        assert client.post("/admin/sleep").status_code == 200
    assert engine.sleeps == [30_000]
    with TestClient(create_app(model, FakeEngine(tokenizer))) as client:
        assert client.post("/admin/sleep").status_code == 404, "no Admin, no admin routes"


def serve(app: Starlette, body: Callable[[int], Awaitable[None]]) -> None:
    async def main() -> None:
        sock = socket.create_server(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning", timeout_graceful_shutdown=2))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            await asyncio.sleep(0.01)
        try:
            await body(port)
        finally:
            server.should_exit = True
            await task

    asyncio.run(main())


async def until(cond: Callable[[], bool], what: str) -> None:
    async with asyncio.timeout(10):
        while not cond():
            await asyncio.sleep(0.01)
    assert cond(), what


def test_the_lease_wakes_the_engine_and_a_sleep_ends_it(model: Model, tokenizer: Tokenizer):
    """llama-swap's cmd (the lease) and cmdStop (lease --sleep), each run as llama-swap would."""
    engine = FakeEngine(tokenizer, status="asleep")

    async def body(port: int) -> None:
        url = f"http://127.0.0.1:{port}"
        held = asyncio.ensure_future(asyncio.to_thread(lease.main, [url]))
        await until(lambda: engine.status == "ready", "the lease wakes the engine")
        assert not held.done(), "the lease holds on"
        assert await asyncio.to_thread(lease.main, [url, "--sleep", "--drain-ms", "100"]) == 0
        assert await held == 0, "a sleep ends the lease, and the lease exits 0"
        assert engine.sleeps == [100]

    serve(create_app(model, engine, Admin(heartbeat_s=0.05)), body)


def test_a_dropped_lease_sleeps_the_engine(model: Model, tokenizer: Tokenizer):
    """Fail closed: the lease's connection ends without a sleep, and the front end sleeps on its own."""
    engine = FakeEngine(tokenizer, status="asleep")

    async def body(port: int) -> None:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /admin/lease HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\n\r\n")
        await writer.drain()
        head = b""
        while b'"ready"' not in head:
            head += await reader.read(4096)
        assert engine.status == "ready"
        writer.close()
        await until(lambda: engine.sleeps == [7000], "the engine sleeps once the last lease is gone")

    serve(create_app(model, engine, Admin(drain_ms=7000, heartbeat_s=0.05)), body)


def test_the_engine_sleeps_only_when_the_last_lease_is_gone(model: Model, tokenizer: Tokenizer):
    engine = FakeEngine(tokenizer, status="asleep")

    async def open_lease(port: int) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"POST /admin/lease HTTP/1.1\r\nhost: x\r\ncontent-length: 0\r\n\r\n")
        head = b""
        while b'"ready"' not in head:
            head += await reader.read(4096)
        return reader, writer

    async def body(port: int) -> None:
        _, first = await open_lease(port)
        _, second = await open_lease(port)
        first.close()
        await asyncio.sleep(0.3)
        assert engine.sleeps == [], "one lease is still held"
        second.close()
        await until(lambda: len(engine.sleeps) == 1, "the last lease gone, the engine sleeps")

    serve(create_app(model, engine, Admin(heartbeat_s=0.05)), body)
