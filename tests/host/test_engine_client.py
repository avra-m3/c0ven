"""The front end's engine client against a real engine process with the mock backend."""

from __future__ import annotations

import asyncio
import sys

import pytest

import protocol
from frontend.engine_client import EngineClient, EngineUnavailable, SleepMode

ARGV = [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", "--slots", "2", "--context", "4096"]
SETTINGS = protocol.Settings(temperature=0.0)


async def run(client: EngineClient, prompt: tuple[int, ...], max_new: int) -> list[protocol.Event]:
    return [e async for e in client.generate(prompt, max_new, SETTINGS)]


def tokens(events: list[protocol.Event]) -> list[int]:
    return [t for e in events if isinstance(e, protocol.Tokens) for t in e.ids]


def test_generate_two_at_once():
    async def main() -> None:
        client = EngineClient(ARGV)
        await client.start()
        assert (client.max_context, client.slots) == (4096, 2)
        assert client.info["backend"] == "mock"
        a, b = await asyncio.gather(run(client, (1, 2, 3), 8), run(client, (4, 5, 6, 7), 8))
        for events in (a, b):
            done = events[-1]
            assert isinstance(done, protocol.Done)
            assert done.generated == len(tokens(events)) > 0
        again = await run(client, (1, 2, 3), 8)
        assert tokens(again) == tokens(a), "greedy, so the same prompt gives the same tokens"
        assert client.busy == 0
        await client.close()
        assert not client.alive

    asyncio.run(main())


def test_leaving_early_stops_the_request():
    async def main() -> None:
        client = EngineClient(ARGV)
        alone = await run(client, (9, 9, 9), 4)
        events = client.generate((1, 2, 3), 3000, SETTINGS)
        async for event in events:
            if isinstance(event, protocol.Tokens):
                break
        await events.aclose()
        assert client.busy == 0
        after = await run(client, (9, 9, 9), 4)
        assert tokens(after) == tokens(alone), "no line of the stopped request reaches a later one"
        await client.close()

    asyncio.run(main())


def test_a_dead_engine_fails_its_requests_and_restarts():
    async def main() -> None:
        client = EngineClient([*ARGV, "--realtime"])
        await client.start()
        process = client._process  # pyright: ignore[reportPrivateUsage]
        assert process is not None

        async def kill_soon() -> None:
            await asyncio.sleep(0.2)
            process.kill()

        events, _ = await asyncio.gather(run(client, (1, 2, 3), 3000), kill_soon())
        assert isinstance(events[-1], protocol.Error)
        assert "exited" in events[-1].message
        assert not client.alive
        events = await run(client, (1, 2, 3), 4)
        assert isinstance(events[-1], protocol.Done)
        assert client.state.starts == 2
        await client.close()

    asyncio.run(main())


def test_asleep_no_request_starts_the_engine_and_a_wake_gives_the_same_tokens():
    """Mode keep: the child stays, SLEEP gives the cards back, WAKE loads them."""

    async def main() -> None:
        client = EngineClient(ARGV)
        await client.start()
        before = await run(client, (1, 2, 3), 8)
        process = client._process  # pyright: ignore[reportPrivateUsage]
        ms = await client.sleep(0)
        assert ms >= 0
        assert (client.status, client.alive) == ("asleep", True)
        with pytest.raises(EngineUnavailable, match="asleep"):
            await run(client, (1, 2, 3), 8)
        info = await client.wake()
        assert client.status == "ready"
        assert "load_s" in info
        assert client._process is process, "the same child, woken"  # pyright: ignore[reportPrivateUsage]
        assert tokens(await run(client, (1, 2, 3), 8)) == tokens(before)
        assert client.state.starts == 1
        await client.close()

    asyncio.run(main())


def test_a_sleep_lets_a_running_request_finish_within_the_drain():
    async def main() -> None:
        client = EngineClient([*ARGV, "--realtime"])
        await client.start()
        request = asyncio.ensure_future(run(client, (1, 2, 3), 20))
        await asyncio.sleep(0.05)
        await client.sleep(30_000)
        done = (await request)[-1]
        assert isinstance(done, protocol.Done)
        assert done.finish is not protocol.Finish.CANCEL
        await client.close()

    asyncio.run(main())


def test_a_sleep_cancels_what_runs_past_the_drain():
    async def main() -> None:
        client = EngineClient([*ARGV, "--realtime"])
        await client.start()
        request = asyncio.ensure_future(run(client, (1, 2, 3), 3000))
        await asyncio.sleep(0.05)
        await client.sleep(50)
        done = (await request)[-1]
        assert isinstance(done, protocol.Done)
        assert done.finish is protocol.Finish.CANCEL
        await client.close()

    asyncio.run(main())


def test_mode_exit_quits_the_child_and_a_wake_starts_another():
    async def main() -> None:
        client = EngineClient(ARGV, mode=SleepMode.EXIT)
        await client.start()
        before = await run(client, (4, 5, 6), 8)
        await client.sleep(0)
        assert (client.status, client.alive) == ("asleep", False)
        await client.wake()
        assert client.state.starts == 2
        assert tokens(await run(client, (4, 5, 6), 8)) == tokens(before)
        await client.close()

    asyncio.run(main())


def test_started_asleep_the_engine_loads_at_the_first_wake():
    async def main() -> None:
        client = EngineClient(ARGV, asleep=True)
        await client.start()
        assert (client.status, client.alive) == ("asleep", False)
        with pytest.raises(EngineUnavailable):
            await run(client, (1, 2, 3), 4)
        await client.wake()
        assert isinstance((await run(client, (1, 2, 3), 4))[-1], protocol.Done)
        await client.close()

    asyncio.run(main())
