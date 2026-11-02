"""tools/check.py's verdicts and request bodies, on runs written in the test; how it ends a run, on a stand-in."""

from __future__ import annotations

import argparse
import json
import socket
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from tools import check
from tools.check import (
    CONFIGS,
    QUICK,
    Check,
    Client,
    Run,
    against_a,
    apart,
    drift,
    far_apart,
    gain,
    logged_error,
    past_stored,
    quick_stop,
    reference_body,
    slower,
)

ROOT = Path(__file__).resolve().parents[2]
CORE_ERROR = "engine.native core error 1: card 0 did not publish layer 23's experts"
# Answers /health and the warm-up. Any other request logs CORE_ERROR when FAKE_ERROR says so, then gets only keepalives.
FAKE_SERVER = """
import argparse, json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        self.send_response(200)
        self.end_headers()
        if "France" in body:
            chunk = {"choices": [{"delta": {"content": "Paris"}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\\n\\ndata: [DONE]\\n\\n".encode())
            return
        print(os.environ.get("FAKE_ERROR", ""), flush=True)
        while True:
            self.wfile.write(b": keepalive\\n\\n")
            self.wfile.flush()
            time.sleep(0.05)

if "FAKE_DEAF" in os.environ:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
p = argparse.ArgumentParser()
p.add_argument("--port", type=int)
ThreadingHTTPServer(("127.0.0.1", p.parse_known_args()[0].port), Handler).serve_forever()
"""


def run(arm: str, seconds: dict[str, float], swapped: int = 0) -> Run:
    r = Run(1, arm, "deployed", tuple(seconds), swapped_pages=swapped)
    r.requests = [{"scenario": s, "seconds": t, "ttft": t / 10, "decode_tps": 50.0} for s, t in seconds.items()]
    return r


def test_apart_needs_every_run_of_one_arm_to_beat_every_run_of_the_other():
    assert apart([10.0, 10.2, 10.1, 10.3], [9.8, 9.9, 9.7, 9.9]) == 1
    assert apart([9.8, 9.9], [10.0, 10.1]) == -1
    assert apart([10.0, 10.2, 10.1, 10.3], [9.8, 10.1, 9.7, 9.9]) == 0


def test_far_apart_takes_ten_times_the_larger_range():
    assert far_apart([133.4, 133.5], [129.0, 129.1]) == 1
    assert far_apart([133.4, 133.5], [133.0, 133.2]) == 0  # 0.35 s apart, under 10 x 0.2 s
    assert far_apart([129.0, 129.0], [133.4, 133.4]) == -1


def test_gain_wants_one_faster_and_none_slower():
    assert gain({"short": 1, "four": 0, "mixed": 0})
    assert not gain({"short": 1, "four": -1, "mixed": 0})
    assert not gain({"short": 0, "four": 0, "mixed": 0})


def test_quick_stop_flags_a_scenario_slower_than_twice_its_spread():
    a1 = run("A", {"short": 6.3, "four": 12.8, "mixed": 16.5})
    assert quick_stop(a1, run("B", {"short": 6.4, "four": 14.0, "mixed": 17.2})) == []  # four's spread is wide
    slow = quick_stop(a1, run("B", {"short": 6.3, "four": 12.8, "mixed": 18.0}))
    assert len(slow) == 1
    assert slow[0].startswith("mixed")
    assert set(QUICK) == {"short", "four", "mixed"}


def test_against_a_fails_twice_the_time_and_a_gib_more_swap():
    a = run("A", {"short": 6.3, "four": 12.8}, swapped=14_499)
    assert against_a(a, run("B", {"short": 6.3, "four": 12.9}, swapped=0)) == []
    assert len(against_a(a, run("B", {"short": 13.0, "four": 12.9}))) == 1
    assert len(against_a(a, run("B", {"short": 6.3, "four": 12.9}, swapped=14_499 + (1 << 18) + 1))) == 1


def test_against_a_leaves_out_a_scenario_a_failed_run_did_not_reach():
    a = run("A", {"short": 6.3, "four": 12.8, "mixed": 16.5})
    b = run("B", {"short": 6.3, "four": 12.9})
    b.scenarios = ("short", "four", "mixed")
    assert against_a(a, b) == []
    assert against_a(b, a) == []


def test_reference_body_tags_the_first_message_and_wraps_tool_calls():
    rows = [json.loads(x) for x in (ROOT / "reference/messages.jsonl").read_text().splitlines() if x.strip()]
    messages = {r["id"]: r for r in rows}
    body = reference_body(messages, "agent-turn2", "four", 512)
    assert body["messages"][0]["content"].startswith("Request four.\n")
    assert body["max_tokens"] == 512
    messages_out: list[dict[str, Any]] = body["messages"]
    calls: list[dict[str, Any]] = []
    for m in messages_out:
        calls += m.get("tool_calls") or []
    assert calls
    assert all(c["type"] == "function" and "function" in c for c in calls)
    assert messages["agent-turn2"]["messages"][0]["content"] in body["messages"][0]["content"]


def test_slower_needs_every_b_run_slower_than_every_a_run():
    a = [run("A", {"short": 6.2, "four": 13.1}), run("A", {"short": 6.2, "four": 13.2})]
    b = [run("B", {"short": 6.4, "four": 12.0}), run("B", {"short": 6.5, "four": 13.5})]
    assert slower(a, b, ("short", "four")) == ["short"]


def test_drift_lets_small_moves_stand_and_names_large_ones():
    then = {
        "mem_avail_gib": 56.0,
        "swap_used_gib": 3.0,
        "cpu_busy": 0.02,
        "governor": "powersave",
        "card0_free_gib": 23.9,
    }
    assert drift(then, then | {"mem_avail_gib": 52.0, "cpu_busy": 0.1}) == []
    moved = drift(then, then | {"card0_free_gib": 1.3, "governor": "performance", "cpu_busy": 0.6})
    assert len(moved) == 3
    assert moved[0].startswith("card0_free_gib")


def test_a_stored_run_gives_the_same_verdict_inputs():
    a = run("A", {"short": 6.2, "four": 13.1}, swapped=7)
    back = Run.from_stored(json.loads(json.dumps(a.stored())))
    assert back.seconds("four") == 13.1
    assert back.first_token("short") == a.first_token("short")
    assert back.swapped_pages == 7


def test_past_stored_allows_the_scenario_spread_for_drift():
    stored = [
        run("A", {"short": 6.19, "four": 12.2, "mixed": 16.4}),
        run("A", {"short": 6.23, "four": 12.6, "mixed": 16.8}),
    ]
    level = [
        run("B", {"short": 6.28, "four": 12.9, "mixed": 16.8}),
        run("B", {"short": 6.26, "four": 12.0, "mixed": 16.9}),
    ]
    assert past_stored(stored, level) == []
    slow = [
        run("B", {"short": 6.40, "four": 12.9, "mixed": 16.8}),
        run("B", {"short": 6.38, "four": 12.0, "mixed": 16.9}),
    ]
    assert past_stored(stored, slow) == ["short"]


def test_every_configuration_keeps_the_ngram_table_in_the_file():
    for flags, _ in CONFIGS.values():
        assert check.NGRAM_IN_FILE in flags


def test_logged_error_gives_the_first_error_line(tmp_path: Path):
    log = tmp_path / "server.log"
    log.write_text("INFO loaded\nINFO 200 OK\n")
    assert logged_error(log) is None
    log.write_text(f"INFO loaded\n{CORE_ERROR}\nERROR the engine failed\n")
    assert logged_error(log) == f"server.log: {CORE_ERROR}"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def no_output(*_: object) -> str:
    return ""


def no_children(_: int) -> list[int]:
    return []


@pytest.fixture
def stand_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Check:
    """A Check whose A arm starts FAKE_SERVER. The limits are short enough that a hung request fails the test soon."""
    tree = tmp_path / "tree"
    (tree / "frontend").mkdir(parents=True)
    (tree / "frontend/__main__.py").write_text(FAKE_SERVER)
    python = tree / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    python.chmod(python.stat().st_mode | stat.S_IEXEC)
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(check, "git", no_output)
    monkeypatch.setattr(check, "head", no_output)
    monkeypatch.setattr(check, "meminfo_gib", lambda: 64.0)
    monkeypatch.setattr(check, "swapped_out", lambda: 0)
    monkeypatch.setattr(check, "descendants", no_children)
    monkeypatch.setattr(check, "FIRST_TOKEN_S", 20.0)
    monkeypatch.setattr(check, "STOP_S", 1.0)
    return Check(
        argparse.Namespace(
            a=tree,
            model=Path("model"),
            out=tmp_path / "out",
            skip=[],
            baseline_key="",
            baselines=None,
            vram_cmd=None,
            port=free_port(),
            pause=0.0,
            mem_floor=22.0,
        )
    )


def timed_run(c: Check) -> tuple[Run, float]:
    t0 = time.monotonic()
    out = c.served("A", "one-slot", ["short"])
    return out, time.monotonic() - t0


def test_a_core_error_ends_the_run_with_a_request_open(stand_in: Check, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_ERROR", CORE_ERROR)
    out, seconds = timed_run(stand_in)
    assert seconds < 15
    assert out.failures[0].endswith(CORE_ERROR)
    assert sum(CORE_ERROR in x for x in out.failures) == 1  # the scan after the run does not report it again
    assert [r["scenario"] for r in out.requests] == ["short"]
    assert out.requests[0]["bad"]


def test_a_core_error_ends_the_run_when_the_server_ignores_sigterm(stand_in: Check, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FAKE_ERROR", CORE_ERROR)
    monkeypatch.setenv("FAKE_DEAF", "1")
    out, seconds = timed_run(stand_in)
    assert seconds < 15
    assert out.failures[0].endswith(CORE_ERROR)
    assert out.requests[0]["bad"]


def test_memory_under_the_floor_ends_the_run_with_a_request_open(stand_in: Check, monkeypatch: pytest.MonkeyPatch):
    levels = iter([64.0, 64.0])
    monkeypatch.setattr(check, "meminfo_gib", lambda: next(levels, 21.5))
    out, seconds = timed_run(stand_in)
    assert seconds < 15
    assert out.failures[0] == "MemAvailable 21.5 GiB, under 22.0 GiB"


def keepalives_after(tokens: int) -> ThreadingHTTPServer:
    """A server whose stream sends `tokens` tokens, then only keepalives for 5 s, so a client that stays fails the test
    and does not hang it."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            chunk = {"choices": [{"delta": {"content": "x"}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode() * tokens)
            end = time.monotonic() + 5
            try:
                while time.monotonic() < end:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    time.sleep(0.02)
            except OSError:
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("tokens", [0, 2])
def test_keepalives_do_not_keep_a_stalled_request_open(monkeypatch: pytest.MonkeyPatch, tokens: int):
    monkeypatch.setattr(check, "FIRST_TOKEN_S", 0.2)
    monkeypatch.setattr(check, "GAP_S", 0.2)
    server = keepalives_after(tokens)
    r = Run(1, "A", "one-slot", ("short",))
    rec = Client(server.server_address[1], r, lambda _: None).send("short", "alone", "id", {"messages": []})
    server.shutdown()
    assert "TimeoutError: no token for" in rec["error"]
    assert r.failures
