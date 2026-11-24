"""Tests for `POST /v1/decide`: exact choice scoring, calibration, swapped-order debiasing, prefix preservation."""

from __future__ import annotations

import io
import math
import threading
from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TextIO, cast

import pytest
from starlette.testclient import TestClient
from tokenizers import Tokenizer

import protocol
from engine.api.mock import MockEngine
from engine.process import SwapWatch, serve
from engine.scheduler import SchedulerOptions, every_16k
from frontend.app import create_app
from frontend.model import Model
from frontend.template import ChatTemplate
from frontend.tokenizer import encode
from tests.helpers import BUDGET_TEMPLATE, LEVEL_TEMPLATE, FakeEngine, open_config


@pytest.fixture
def engine(tokenizer: Tokenizer) -> FakeEngine:
    return FakeEngine(tokenizer)


@pytest.fixture
def client(model: Model, engine: FakeEngine) -> Iterator[TestClient]:
    with TestClient(create_app(model, engine)) as c:
        yield c


def test_decide_returns_exact_probabilities_including_below_point_one_percent(
    client: TestClient, engine: FakeEngine, tokenizer: Tokenizer
):
    # Choice "3" has logit -2.0 vs +10.0 -> ~6.1e-6 (0.0006%, well outside any top-64 cutoff).
    engine.score_fn = lambda _p, _c: ((10.0, 7.0, -2.0), 10.05)
    engine.reused = 32000
    r = client.post(
        "/v1/decide",
        json={
            "messages": [{"role": "user", "content": "Run rm -rf /tmp/x?"}],
            "check_prompt": "Is this command safe?",
            "choices": ["Allow", "Ask", "Block"],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["object"] == "decision"
    assert body["choice"] == "Allow"
    probs = body["probabilities"]
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["Block"] == pytest.approx(math.exp(-12.0) / (1.0 + math.exp(-3.0) + math.exp(-12.0)), rel=1e-6)
    assert 0.0 < probs["Block"] < 0.001
    assert body["logits"] == {"Allow": 10.0, "Ask": 7.0, "Block": -2.0}
    assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == 32000
    assert body["swapped"] is False
    # Auto-assigned single-token keys "1", "2", "3".
    assert engine.decide_choices == [[encode(tokenizer, k)[0] for k in ("1", "2", "3")]]


def test_decide_temperature_scales_probabilities(client: TestClient, engine: FakeEngine):
    engine.score_fn = lambda _p, _c: ((4.0, 2.0), 4.2)
    r1 = client.post(
        "/v1/decide",
        json={
            "messages": [{"role": "user", "content": "check"}],
            "choices": [{"label": "yes", "token": "1"}, {"label": "no", "token": "2"}],
            "temperature": 1.0,
        },
    )
    r2 = client.post(
        "/v1/decide",
        json={
            "messages": [{"role": "user", "content": "check"}],
            "choices": [{"label": "yes", "token": "1"}, {"label": "no", "token": "2"}],
            "temperature": 2.0,
        },
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    p1 = r1.json()["probabilities"]["yes"]
    p2 = r2.json()["probabilities"]["yes"]
    assert p1 == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert p2 == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert p2 < p1


def test_decide_swap_order_averages_both_orderings_and_cancels_position_bias(
    client: TestClient, engine: FakeEngine, tokenizer: Tokenizer
):
    # Simulate a pure first-position bias: key "1" always gets logit 3.0, key "2" always gets logit 1.0.
    def biased(_prompt: Sequence[int], _choices: Sequence[int]) -> tuple[Sequence[float], float]:
        return (3.0, 1.0), 3.15

    engine.score_fn = biased
    r = client.post(
        "/v1/decide",
        json={
            "messages": [{"role": "user", "content": "Run command?"}],
            "check_prompt": "Should we allow?",
            "choices": ["Allow", "Block"],
            "swap_order": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["swapped"] is True
    assert len(engine.prompts) == 2
    assert body["probabilities"]["Allow"] == pytest.approx(0.5)
    assert body["probabilities"]["Block"] == pytest.approx(0.5)
    assert len(body["passes"]) == 2
    assert body["passes"][0]["order"] == ["Allow", "Block"]
    assert body["passes"][1]["order"] == ["Block", "Allow"]
    # Both rendered prompts share the exact prefix before the check message.
    p0 = tokenizer.decode(engine.prompts[0], skip_special_tokens=False)
    p1 = tokenizer.decode(engine.prompts[1], skip_special_tokens=False)
    assert "1: Allow\n2: Block" in p0
    assert "1: Block\n2: Allow" in p1


def test_decide_swap_order_keeps_explicit_caller_tokens_bound_to_their_choices(
    client: TestClient, engine: FakeEngine, tokenizer: Tokenizer
):
    tok_y = encode(tokenizer, "Y")[0]
    tok_n = encode(tokenizer, "N")[0]

    # Always favour the first choice passed in the prompt's presentation order.
    def first_biased(_prompt: Sequence[int], choices: Sequence[int]) -> tuple[Sequence[float], float]:
        return (4.0, 2.0), 4.15

    engine.score_fn = first_biased
    r = client.post(
        "/v1/decide",
        json={
            "messages": [{"role": "user", "content": "Run command?"}],
            "check_prompt": "Allow this action?",
            "choices": [{"label": "allow", "token": "Y"}, {"label": "block", "token": "N"}],
            "swap_order": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["probabilities"]["allow"] == pytest.approx(0.5)
    assert body["probabilities"]["block"] == pytest.approx(0.5)
    assert engine.decide_choices == [[tok_y, tok_n], [tok_n, tok_y]]
    p0 = tokenizer.decode(engine.prompts[0], skip_special_tokens=False)
    p1 = tokenizer.decode(engine.prompts[1], skip_special_tokens=False)
    assert "Y: allow\nN: block" in p0
    assert "N: block\nY: allow" in p1


@pytest.mark.parametrize("template", [LEVEL_TEMPLATE, BUDGET_TEMPLATE])
def test_decide_preserves_thinking_prefix_and_closes_think_at_tail(
    model: Model, engine: FakeEngine, tokenizer: Tokenizer, template: str
):
    """Closing `<think>` at the tail leaves the conversation prefix identical to a thinking chat turn so checkpoints
    taken during the live session stay valid."""
    with TestClient(create_app(replace(model, template=ChatTemplate(template)), engine)) as c:
        messages = [{"role": "user", "content": "Inspect repo"}]
        r_chat = c.post("/v1/chat/completions", json={"messages": messages})
        assert r_chat.status_code == 200
        chat_prompt = tokenizer.decode(engine.prompts[0], skip_special_tokens=False)

        r_dec = c.post(
            "/v1/decide",
            json={
                "messages": [*messages, {"role": "assistant", "content": "Running ls"}],
                "check_prompt": "Safe?",
                "choices": ["Allow", "Block"],
            },
        )
        assert r_dec.status_code == 200, r_dec.text
        decide_prompt = tokenizer.decode(engine.prompts[1], skip_special_tokens=False)

    # Everything up to the first assistant turn matches byte for byte (including the leading effort sentence).
    cut = chat_prompt.index("<|im_start|>assistant")
    assert decide_prompt[:cut] == chat_prompt[:cut]
    assert decide_prompt.endswith(("</think>\n\n", "</think>"))


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [], "choices": ["1", "2"]},
        {"messages": [{"role": "user", "content": "x"}], "choices": []},
        {"messages": [{"role": "user", "content": "x"}], "choices": ["ab", "cd"]},  # multi-token without check_prompt
        {"messages": [{"role": "user", "content": "x"}], "choices": ["1", "1"]},  # duplicate label
        {"messages": [{"role": "user", "content": "x"}], "choices": ["1", "2"], "temperature": 0},
        {"messages": [{"role": "user", "content": "x"}], "choices": ["1", "2"], "swap_order": True},  # no check_prompt
        {
            "messages": [{"role": "user", "content": "x"}],
            "check_prompt": "Pick one",
            "choices": ["OnlyOne"],
            "swap_order": True,
        },
        {"messages": [{"role": "user", "content": "x"}], "choices": [{"label": "a", "token_id": 999999}]},
    ],
)
def test_decide_bad_requests(client: TestClient, body: dict[str, object]):
    assert client.post("/v1/decide", json=body).status_code == 400


def test_decide_refused_with_503_while_asleep(client: TestClient, engine: FakeEngine):
    engine.status = "asleep"
    r = client.post(
        "/v1/decide",
        json={"messages": [{"role": "user", "content": "hi"}], "choices": ["1", "2"]},
    )
    assert r.status_code == 503


def test_engine_process_serves_decide_over_pipe(tmp_path: Path):
    """End-to-end line protocol through engine.process.serve(): DECIDE emits SCORES then DONE."""
    done_ev = threading.Event()
    lines: list[str] = []

    class Out(io.StringIO):
        def write(self, s: str) -> int:
            lines.append(s)
            if "DONE 7 " in s:
                done_ev.set()
            return super().write(s)

    def feed() -> Iterator[str]:
        yield protocol.Decide(7, (10, 20, 30), (3, 4, 5, 6)).encode() + "\n"
        done_ev.wait(timeout=5.0)
        yield protocol.Quit().encode() + "\n"

    out = Out()
    serve(
        MockEngine(),
        open_config(),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=cast("TextIO", feed()),
        stdout=out,
        swap_watch=SwapWatch(tmp_path / "none"),
    )
    events = [protocol.parse_event(line) for line in out.getvalue().splitlines()]
    (sc,) = [e for e in events if isinstance(e, protocol.Scores) and e.rid == 7]
    (done,) = [e for e in events if isinstance(e, protocol.Done) and e.rid == 7]
    assert len(sc.logits) == 3
    assert done.generated == 0
    assert done.finish is protocol.Finish.STOP
