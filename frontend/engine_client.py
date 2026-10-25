"""The front end's side of the engine: the child process and the line protocol (DESIGN.md, section 1).

The front end starts the engine as its child and talks to it in token ids. Lines come back tagged with request ids
and are handed to each request's queue. If the engine dies, every request in flight gets an error, and the next
request starts it again, as Strata's server does (spec/STRATA_DECODE.md, section 10).

A request's images go before its GEN line: each image's patches in a memfd, named by an IMG line as
/proc/<pid>/fd/<n>, which the engine reads when it reads the line. The memfd is closed when the request ends.

The engine sleeps and wakes on request (ADR-0032). In mode `keep` the child stays and gives the cards
back (`SLEEP`), keeping the arena in host memory, and `WAKE` loads the cards again; in mode `exit` it quits after the
drain, and a wake starts it. While asleep, from the sleep's start until the wake's `READY`, no request starts it:
each is refused with EngineUnavailable.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import os
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

import protocol
from frontend.images import PromptImage

log = logging.getLogger("frontend.engine")

LINE_LIMIT = 16 * 2**20
"""The longest line read: a 262K-token prompt is ~2 MB of ids as text."""


class EngineUnavailable(RuntimeError):
    pass


class SleepMode(StrEnum):
    KEEP = "keep"
    """The child sleeps: the cards are freed, the arena and the child's runtime stay."""
    EXIT = "exit"
    """The child exits: the cards and the host memory go back, and a wake loads everything again."""


@dataclass(slots=True)
class EngineState:
    info: dict[str, str] = field(default_factory=lambda: {})
    max_context: int = 0
    slots: int = 0
    starts: int = 0
    running: dict[int, int] = field(default_factory=lambda: {})
    """Requests in flight: id to tokens generated so far."""


class EngineClient:
    def __init__(
        self,
        argv: Sequence[str],
        start_timeout_s: float = 600.0,
        mode: SleepMode = SleepMode.KEEP,
        asleep: bool = False,
    ) -> None:
        self.argv = list(argv)
        self.start_timeout_s = start_timeout_s
        self.mode = mode
        self.state = EngineState()
        self._asleep = asleep
        """From a sleep's start to a wake's READY: no request starts the engine."""
        self._waking = False
        self._sleeping = False
        self._control: asyncio.Queue[protocol.Event] = asyncio.Queue()
        """The lines about the engine, not a request: INFO, READY, ASLEEP, and ERR - or the child's exit."""
        self._stats: asyncio.Queue[protocol.StatsReport] = asyncio.Queue()
        self._stats_lock = asyncio.Lock()
        """One STATS question at a time, so each answer is its own."""
        self._process: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task[None] | None = None
        self._queues: dict[int, asyncio.Queue[protocol.Event]] = {}
        self._ids = itertools.count(1)
        self._lock = asyncio.Lock()
        self._write_lock = asyncio.Lock()
        self._stopping: set[asyncio.Task[None]] = set()
        """Requests left early, draining; held here so the loop does not drop them."""

    @property
    def max_context(self) -> int:
        return self.state.max_context

    @property
    def info(self) -> dict[str, str]:
        return self.state.info

    @property
    def image_tokens_max(self) -> int:
        """The most tokens an image may have; 0 when the engine reads no images (or has not started)."""
        try:
            return int(self.state.info.get("image_tokens_max", "0"))
        except ValueError:
            return 0

    @property
    def slots(self) -> int:
        return self.state.slots

    @property
    def busy(self) -> int:
        return len(self.state.running)

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def asleep(self) -> bool:
        return self._asleep

    @property
    def status(self) -> str:
        """ready, stopped (the next request starts it), loading (a wake), sleeping (the drain), or asleep."""
        if self._waking:
            return "loading"
        if self._sleeping:
            return "sleeping"
        if self._asleep:
            return "asleep"
        return "ready" if self.alive else "stopped"

    async def start(self) -> None:
        async with self._lock:
            if self.alive or self._asleep:
                return
            await self._spawn()

    async def wake(self) -> dict[str, str]:
        """Loads the engine and returns its INFO, once it is READY. A child asleep is sent WAKE; with none, one is
        started. On failure the engine stays asleep, and a child that did not wake in time is killed."""
        async with self._lock:
            if self.alive and not self._asleep:
                return self.state.info
            self._waking = True
            try:
                if self.alive:
                    await self._wake_child()
                else:
                    await self._spawn()
            finally:
                self._waking = False
            self._asleep = False
            return self.state.info

    async def _wake_child(self) -> None:
        self._drain_control()
        await self._send(protocol.Wake().encode())
        info: dict[str, str] = {}
        try:
            async with asyncio.timeout(self.start_timeout_s):
                while True:
                    match await self._control.get():
                        case protocol.Info(facts=facts):
                            info.update(facts)
                        case protocol.Ready(max_context=ctx, slots=slots):
                            self.state.max_context, self.state.slots = ctx, slots
                            break
                        case protocol.Error(message=message):
                            raise EngineUnavailable(f"the engine failed to wake: {message}")
                        case _:
                            pass
        except TimeoutError:
            await self._kill()
            raise EngineUnavailable(f"the engine did not wake within {self.start_timeout_s:g} s") from None
        self.state.info = info

    async def sleep(self, drain_ms: int) -> float:
        """Stops admitting, lets the running requests go on for up to drain_ms, cancels the rest, and gives the cards
        back: SLEEP, then in mode exit QUIT. Returns the milliseconds the engine took. A child that does not answer
        in time is killed, so the cards are free when this returns."""
        self._asleep = True
        async with self._lock:
            if not self.alive:
                return 0.0
            self._sleeping = True
            try:
                ms = await self._sleep_child(drain_ms)
                if self.mode is SleepMode.EXIT:
                    await self._quit()
            finally:
                self._sleeping = False
            return ms

    async def _sleep_child(self, drain_ms: int) -> float:
        self._drain_control()
        try:
            await self._send(protocol.Sleep(drain_ms).encode())
            async with asyncio.timeout(drain_ms / 1e3 + 60):
                while True:
                    match await self._control.get():
                        case protocol.Asleep(ms=ms):
                            return ms
                        case protocol.Error():
                            return 0.0  # the child is gone, and the cards with it
                        case _:
                            pass
        except TimeoutError, EngineUnavailable, ConnectionError:
            log.error("the engine did not sleep; killing it")
            await self._kill()
            return 0.0

    def _drain_control(self) -> None:
        while not self._control.empty():
            self._control.get_nowait()

    async def _spawn(self) -> None:
        log.info("starting the engine: %s", " ".join(self.argv))
        process = await asyncio.create_subprocess_exec(
            *self.argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, limit=LINE_LIMIT
        )
        assert process.stdout is not None
        state = self.state
        state.info = {}
        try:
            async with asyncio.timeout(self.start_timeout_s):
                while True:
                    line = await process.stdout.readline()
                    if not line:
                        raise EngineUnavailable(f"the engine exited while starting ({await process.wait()})")
                    event = protocol.parse_event(line.decode())
                    match event:
                        case protocol.Info(facts=facts):
                            state.info.update(facts)
                        case protocol.Ready(max_context=ctx, slots=slots):
                            state.max_context, state.slots = ctx, slots
                            break
                        case protocol.Error(message=message):
                            raise EngineUnavailable(f"the engine failed to start: {message}")
                        case _:
                            raise EngineUnavailable(f"unexpected line while starting: {line[:80]!r}")
        except BaseException:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            raise
        state.starts += 1
        self._process = process
        self._reader = asyncio.create_task(self._read(process), name="engine-reader")

    async def _read(self, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None
        try:
            while line := await process.stdout.readline():
                try:
                    event = protocol.parse_event(line.decode())
                except protocol.ProtocolError:
                    log.exception("bad line from the engine")
                    continue
                rid = getattr(event, "rid", None)
                if isinstance(event, protocol.Error) and rid is None:
                    log.error("the engine failed: %s", event.message)
                    self._control.put_nowait(event)
                    break
                if isinstance(event, protocol.StatsReport):
                    self._stats.put_nowait(event)
                elif isinstance(event, (protocol.Info, protocol.Ready, protocol.Asleep)):
                    self._control.put_nowait(event)
                elif isinstance(rid, int) and (q := self._queues.get(rid)) is not None:
                    q.put_nowait(event)
        finally:
            code = await process.wait()
            log.warning("the engine exited with %s", code)
            self._control.put_nowait(protocol.Error(None, f"the engine exited ({code})"))
            for rid, q in self._queues.items():
                q.put_nowait(protocol.Error(rid, f"the engine exited ({code})"))

    async def stats(self) -> dict[str, str]:
        """The engine's counters since it started (STATS): the core's and the scheduler's. Empty when no child runs,
        or none answers within 5 s. Never starts the engine."""
        if not self.alive:
            return {}
        async with self._stats_lock:
            while not self._stats.empty():
                self._stats.get_nowait()
            try:
                await self._send(protocol.Stats().encode())
                async with asyncio.timeout(5):
                    return (await self._stats.get()).counts
            except EngineUnavailable, ConnectionError, TimeoutError:
                return {}

    async def _send(self, line: str) -> None:
        process = self._process
        if process is None or process.stdin is None or not self.alive:
            raise EngineUnavailable("the engine is not running")
        async with self._write_lock:
            process.stdin.write(line.encode() + b"\n")
            await process.stdin.drain()

    async def generate(
        self,
        prompt: Sequence[int],
        max_new: int,
        settings: protocol.Settings,
        images: Sequence[PromptImage] = (),
    ) -> AsyncGenerator[protocol.Event]:
        """The request's events, ending with Done or Error. Leaving early (a client gone, a stop string) sends
        STOP and drains the request's lines in the background, so they never reach a later request."""
        if self._asleep:
            raise EngineUnavailable("the engine is asleep")
        await self.start()
        rid = next(self._ids)
        q: asyncio.Queue[protocol.Event] = asyncio.Queue()
        self._queues[rid] = q
        self.state.running[rid] = 0
        finished = False
        fds: list[int] = []
        try:
            for im in images:
                fd = os.memfd_create(f"image-{rid}", os.MFD_CLOEXEC)
                fds.append(fd)
                view = memoryview(im.pixels)
                while view:
                    view = view[os.write(fd, view) :]
                path = f"/proc/{os.getpid()}/fd/{fd}"
                await self._send(protocol.Img(rid, im.first, im.grid_h, im.grid_w, path, im.frames).encode())
            await self._send(protocol.Gen(rid, max_new, tuple(prompt), settings).encode())
            while True:
                event = await q.get()
                if isinstance(event, protocol.Tokens):
                    self.state.running[rid] += len(event.ids)
                if isinstance(event, (protocol.Done, protocol.Error)):
                    finished = True
                yield event
                if finished:
                    return
        finally:
            for fd in fds:
                os.close(fd)
            self.state.running.pop(rid, None)
            if finished:
                self._queues.pop(rid, None)
            else:
                task = asyncio.get_running_loop().create_task(self._stop(rid, q))
                self._stopping.add(task)
                task.add_done_callback(self._stopping.discard)

    async def _stop(self, rid: int, q: asyncio.Queue[protocol.Event]) -> None:
        try:
            with contextlib.suppress(EngineUnavailable):
                await self._send(protocol.Stop(rid).encode())
            async with asyncio.timeout(60):
                while not isinstance(await q.get(), (protocol.Done, protocol.Error)):
                    pass
        except TimeoutError:
            log.error("request %d did not finish after STOP", rid)
        finally:
            self._queues.pop(rid, None)

    async def close(self) -> None:
        if self._process is None:
            return
        if self._stopping:
            await asyncio.gather(*self._stopping)
        await self._quit()

    async def _quit(self) -> None:
        process = self._process
        if process is None:
            return
        with contextlib.suppress(EngineUnavailable, ConnectionError):
            await self._send(protocol.Quit().encode())
        try:
            async with asyncio.timeout(10):
                await process.wait()
        except TimeoutError:
            process.kill()
            await process.wait()
        if self._reader is not None:
            await self._reader
        self._process = None

    async def _kill(self) -> None:
        process = self._process
        if process is None:
            return
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()
        if self._reader is not None:
            await self._reader
        self._process = None
