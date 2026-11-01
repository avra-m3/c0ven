"""tools/check.py's verdicts and request bodies, on runs written in the test."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.check import (
    QUICK,
    Run,
    against_a,
    apart,
    drift,
    far_apart,
    gain,
    past_stored,
    quick_stop,
    reference_body,
    slower,
)

ROOT = Path(__file__).resolve().parents[2]


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
