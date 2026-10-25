"""The engine process over its pipe, with the mock behind it."""

from __future__ import annotations

import io
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import TextIO, cast

import protocol
from engine.api.interface import StepPlan
from engine.api.mock import MockEngine
from engine.process import serve
from engine.scheduler import SchedulerOptions, every_16k
from tests.helpers import open_config


def run_engine(lines: list[str], *args: str) -> list[protocol.Event]:
    """Send the lines, read until each request has ended, then QUIT. (QUIT, or the pipe closing, ends the engine at
    once, as when the front end goes away.)"""
    argv = [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", *args]
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdin is not None
    assert p.stdout is not None
    for line in lines:
        p.stdin.write(line + "\n")
    p.stdin.flush()
    events: list[protocol.Event] = []
    ended = 0
    while ended < len(lines):
        line = p.stdout.readline()
        assert line, "the engine closed its output"
        events.append(protocol.parse_event(line))
        ended += isinstance(events[-1], (protocol.Done, protocol.Error))
    p.stdin.write("QUIT\n")
    p.stdin.close()
    assert p.wait(timeout=30) == 0
    return events


def test_info_ready_and_requests():
    events = run_engine(
        [protocol.Gen(1, 3, (1, 2, 3)).encode(), protocol.Gen(2, 2, (4, 5)).encode()],
        "--slots",
        "2",
        "--context",
        "1024",
    )
    assert isinstance(events[0], protocol.Info)
    assert events[0].facts["slots"] == "2"
    assert events[1] == protocol.Ready(1024, 2)
    done = {e.rid: e for e in events if isinstance(e, protocol.Done)}
    assert sorted(done) == [1, 2]
    tokens = {
        rid: [t for e in events if isinstance(e, protocol.Tokens) and e.rid == rid for t in e.ids] for rid in done
    }
    assert [len(tokens[1]), len(tokens[2])] == [done[1].generated, done[2].generated]
    assert all(d.generated <= 3 for d in done.values())


def test_a_request_too_long_is_refused_and_others_go_on():
    events = run_engine([protocol.Gen(1, 5000, (1,)).encode(), protocol.Gen(2, 1, (7,)).encode()], "--context", "512")
    assert protocol.Error(1, "prompt (1) plus max_new (5000) exceeds the context (512)") in events
    assert any(isinstance(e, protocol.Done) and e.rid == 2 for e in events)


def test_a_plan_that_does_not_fit_is_refused():
    p = subprocess.run(
        [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", "--slots", "64"],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert p.returncode == 1
    assert p.stdout.startswith("ERR - memory plan:")


class Pipe:
    """The engine process, one line at a time."""

    def __init__(self, *args: str) -> None:
        argv = [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", *args]
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def send(self, command: protocol.Command) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(command.encode() + "\n")
        self.p.stdin.flush()

    def until(self, kind: type[protocol.Event]) -> list[protocol.Event]:
        """The events up to and with the first of kind."""
        assert self.p.stdout is not None
        out: list[protocol.Event] = []
        while not out or not isinstance(out[-1], kind):
            line = self.p.stdout.readline()
            assert line, "the engine closed its output"
            out.append(protocol.parse_event(line))
        return out

    def quit(self) -> int:
        self.send(protocol.Quit())
        return self.p.wait(timeout=30)


def test_asleep_the_engine_refuses_requests_and_after_wake_it_serves_again():
    e = Pipe("--context", "1024")
    first = e.until(protocol.Ready)[0]
    assert isinstance(first, protocol.Info)
    e.send(protocol.Gen(1, 4, (1, 2, 3)))
    assert isinstance(e.until(protocol.Done)[-1], protocol.Done)
    e.send(protocol.Sleep())
    assert isinstance(e.until(protocol.Asleep)[-1], protocol.Asleep)
    e.send(protocol.Gen(2, 4, (1, 2, 3)))
    assert e.until(protocol.Error)[-1] == protocol.Error(2, "the engine is asleep")
    e.send(protocol.Wake())
    woke = e.until(protocol.Ready)
    info = woke[0]
    assert isinstance(info, protocol.Info)
    assert info.facts.keys() == first.facts.keys()
    assert woke[-1] == protocol.Ready(1024, 1)
    e.send(protocol.Gen(3, 4, (1, 2, 3)))
    done = e.until(protocol.Done)[-1]
    assert isinstance(done, protocol.Done)
    assert done.rid == 3
    assert e.quit() == 0


def test_images_sent_while_asleep_go_with_their_refused_request():
    """An IMG whose file is missing, then its GEN, while asleep: the GEN is refused for the sleep, and the image does
    not wait for a later request of the same id."""
    e = Pipe("--context", "1024")
    e.until(protocol.Ready)
    e.send(protocol.Sleep())
    e.until(protocol.Asleep)
    e.send(protocol.Img(2, 0, 2, 2, "/nonexistent/image"))
    e.send(protocol.Gen(2, 4, (1, 2, 3)))
    assert e.until(protocol.Error)[-1] == protocol.Error(2, "the engine is asleep")
    e.send(protocol.Wake())
    e.until(protocol.Ready)
    e.send(protocol.Gen(2, 4, (1, 2, 3)))
    e.send(protocol.Gen(3, 4, (1, 2, 3)))
    events = e.until(protocol.Done)
    assert not [x for x in events if isinstance(x, protocol.Error)]
    done = events[-1]
    assert isinstance(done, protocol.Done)
    assert done.rid == 2
    assert e.quit() == 0


def test_sleep_cancels_the_running_request_and_refuses_the_queued_one():
    """One slot: request 1 runs (the mock taking 11 ms a step), request 2 waits for the slot."""
    e = Pipe("--context", "8192", "--realtime")
    e.until(protocol.Ready)
    e.send(protocol.Gen(1, 4000, (1, 2, 3)))
    e.send(protocol.Gen(2, 4, (4, 5)))
    e.until(protocol.Tokens)
    e.send(protocol.Sleep(drain_ms=50))
    events = e.until(protocol.Asleep)
    assert protocol.Error(2, "the engine is going to sleep") in events
    (done,) = [x for x in events if isinstance(x, protocol.Done)]
    assert done.rid == 1
    assert done.finish is protocol.Finish.CANCEL
    assert done.generated < 4000
    asleep = events[-1]
    assert isinstance(asleep, protocol.Asleep)
    assert asleep.ms >= 50
    assert e.quit() == 0


def test_stats_are_answered_awake_and_asleep():
    e = Pipe("--context", "1024")
    e.until(protocol.Ready)
    e.send(protocol.Gen(1, 3, (1, 2, 3)))
    e.until(protocol.Done)
    e.send(protocol.Stats())
    report = e.until(protocol.StatsReport)[-1]
    assert isinstance(report, protocol.StatsReport)
    assert int(report.counts["plans"]) > 0
    assert int(report.counts["steps"]) >= 2
    assert report.counts["writing"] == "0"
    e.send(protocol.Sleep())
    e.until(protocol.Asleep)
    e.send(protocol.Stats())
    asleep = e.until(protocol.StatsReport)[-1]
    assert isinstance(asleep, protocol.StatsReport)
    assert asleep.counts["plans"] == report.counts["plans"]
    assert e.quit() == 0


def test_the_load_facts_are_in_the_info_line():
    e = Pipe()
    info = e.until(protocol.Ready)[0]
    assert isinstance(info, protocol.Info)
    assert float(info.facts["load_s"]) >= 0
    assert {"pswpout", "compact_stall"} <= info.facts.keys()
    assert e.quit() == 0


class Recording(MockEngine):
    def __init__(self) -> None:
        super().__init__()
        self.plans: list[StepPlan] = []

    def submit(self, plan: StepPlan) -> None:
        self.plans.append(plan)
        super().submit(plan)


def lines_apart(lines: list[str], apart_s: float, quit_after_s: float) -> Iterator[str]:
    """A front end writing its lines one by one, then QUIT."""
    for line in lines:
        time.sleep(apart_s)
        yield line + "\n"
    time.sleep(quit_after_s)
    yield protocol.Quit().encode() + "\n"


def test_requests_sent_together_read_their_prompts_in_one_plan():
    engine = Recording()
    gens = [protocol.Gen(rid, 4, tuple(range(3, 11))).encode() for rid in range(1, 5)]
    stdin = cast("TextIO", lines_apart(gens, apart_s=0.0002, quit_after_s=0.2))
    status = serve(
        engine,
        open_config(slots=4),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=stdin,
        stdout=io.StringIO(),
    )
    assert status == 0
    first = next(p for p in engine.plans if p.prompt)
    assert sorted(span.slot for span in first.prompt) == [0, 1, 2, 3]
