"""The two HTTP APIs, with a fake engine behind them."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import pytest
from starlette.testclient import TestClient
from starlette.types import Message, Scope
from tokenizers import Tokenizer

from frontend.app import create_app
from frontend.model import Model
from tests.helpers import FakeEngine

CALL = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>"
TOOL = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}},
}


@pytest.fixture
def engine(tokenizer: Tokenizer) -> FakeEngine:
    return FakeEngine(tokenizer)


@pytest.fixture
def client(model: Model, engine: FakeEngine) -> Iterator[TestClient]:
    with TestClient(create_app(model, engine)) as c:
        yield c


def sse(text: str) -> list[tuple[str | None, Any]]:
    out: list[tuple[str | None, Any]] = []
    for block in text.strip().split("\n\n"):
        event = None
        data = None
        for line in block.splitlines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        assert data is not None
        out.append((event, None if data == "[DONE]" else json.loads(data)))
    return out


def chat(client: TestClient, **body: object) -> Any:
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], **body})
    assert r.status_code == 200, r.text
    return r.json()


def test_openai_reasoning_and_content(client: TestClient, engine: FakeEngine):
    engine.output = "Let me think.\n</think>\n\nHello there."
    body = chat(client)
    choice = body["choices"][0]
    assert choice["message"] == {"role": "assistant", "content": "Hello there.", "reasoning_content": "Let me think."}
    assert choice["finish_reason"] == "stop"
    usage = body["usage"]
    assert usage["prompt_tokens"] == len(engine.prompts[0])
    # The fake engine reports a prompt of 1 ms and a decode of 2 ms.
    assert usage["prompt_tokens_per_sec"] == len(engine.prompts[0]) * 1000
    assert usage["completion_tokens_per_sec"] == (usage["completion_tokens"] - 1) * 500
    assert usage["total_time"] == 0.003


def test_openai_thinking_off_sends_the_closed_block(client: TestClient, engine: FakeEngine, tokenizer: Tokenizer):
    engine.output = "Hello."
    body = chat(client, chat_template_kwargs={"enable_thinking": False})
    assert body["choices"][0]["message"]["content"] == "Hello."
    assert tokenizer.decode(engine.prompts[0], skip_special_tokens=False).endswith("<think>\n\n</think>\n\n")


def test_openai_tool_calls(client: TestClient, engine: FakeEngine):
    engine.output = CALL
    body = chat(client, tools=[TOOL], chat_template_kwargs={"enable_thinking": False})
    choice = body["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    (call,) = choice["message"]["tool_calls"]
    assert call["function"] == {"name": "get_weather", "arguments": '{"city": "Paris"}'}


def test_openai_assistant_tool_call_history_renders(client: TestClient, engine: FakeEngine, tokenizer: Tokenizer):
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "rain"},
    ]
    r = client.post("/v1/chat/completions", json={"messages": messages, "tools": [TOOL]})
    assert r.status_code == 200, r.text
    prompt = tokenizer.decode(engine.prompts[0], skip_special_tokens=False)
    assert "<function=get_weather>\n<parameter=city>\nParis\n</parameter>" in prompt


def test_openai_stream(client: TestClient, engine: FakeEngine):
    engine.output = "Hmm.\n</think>\n\nHi!"
    r = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    assert r.status_code == 200
    events = sse(r.text)
    assert events[-1] == (None, None)
    chunks = [d for _, d in events[:-1]]
    deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
    assert "".join(d.get("reasoning_content", "") for d in deltas) == "Hmm."
    assert "".join(d.get("content", "") for d in deltas) == "Hi!"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop"
    assert chunks[-1]["usage"]["completion_tokens"] > 0
    assert chunks[-1]["usage"]["total_time"] == 0.003


def test_openai_stop_string_stops_the_engine(client: TestClient, engine: FakeEngine):
    engine.output = "one two STOP three"
    body = chat(client, stop=["STOP"], chat_template_kwargs={"enable_thinking": False})
    assert body["choices"][0]["message"]["content"] == "one two "
    assert engine.stopped_early == 1


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/v1/chat/completions", {}),
        ("/v1/messages", {"max_tokens": 64}),
    ],
)
def test_a_client_gone_stops_its_request(model: Model, engine: FakeEngine, path: str, body: dict[str, object]):
    engine.hang = True
    app = create_app(model, engine)
    messages: list[Message] = [
        {"type": "http.request", "body": json.dumps({"messages": [{"role": "user", "content": "hi"}], **body}).encode()}
    ]
    gone = asyncio.Event()
    statuses: list[int] = []

    async def receive() -> Message:
        if messages:
            return messages.pop()
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            statuses.append(message["status"])

    async def main() -> None:
        scope: Scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [(b"content-type", b"application/json")],
        }
        call = asyncio.create_task(app(scope, receive, send))
        while not engine.prompts:
            await asyncio.sleep(0)
        gone.set()
        await asyncio.wait_for(call, 5)

    asyncio.run(main())
    assert engine.stopped_early == 1
    assert statuses == [499]


def test_openai_cuts_max_tokens_to_the_context(client: TestClient, engine: FakeEngine):
    engine.max_context = 80
    r = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 30}], "max_tokens": 80}
    )
    assert r.status_code == 200
    assert len(engine.prompts) == 1
    assert engine.max_news == [80 - len(engine.prompts[0])], "the rest of the context"


def test_openai_refuses_a_prompt_that_fills_the_context(client: TestClient, engine: FakeEngine):
    engine.max_context = 20
    r = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 30}], "max_tokens": 20}
    )
    assert r.status_code == 400
    assert "fills the context" in r.json()["error"]["message"]
    assert engine.prompts == [], "refused, never truncated"


@pytest.mark.parametrize(
    "body",
    [
        {"messages": []},
        {"messages": [{"role": "wizard", "content": "x"}]},
        {"messages": [{"role": "user", "content": "x"}], "top_k": 0},
        {"messages": [{"role": "user", "content": "x"}], "temperature": "hot"},
        {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
        {"messages": [{"role": "user", "content": "x"}], "reasoning_effort": "max"},
    ],
)
def test_openai_bad_requests(client: TestClient, body: dict[str, object]):
    assert client.post("/v1/chat/completions", json=body).status_code == 400


def test_openai_json_output_is_checked(client: TestClient, engine: FakeEngine, tokenizer: Tokenizer):
    engine.output = '{"a": 1}'
    fmt = {"response_format": {"type": "json_object"}, "chat_template_kwargs": {"enable_thinking": False}}
    assert chat(client, **fmt)["choices"][0]["message"]["content"] == '{"a": 1}'
    assert "single JSON value" in tokenizer.decode(engine.prompts[0], skip_special_tokens=False)
    engine.output = "not json"
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], **fmt})
    assert r.status_code == 400


def test_anthropic_message(client: TestClient, engine: FakeEngine, tokenizer: Tokenizer):
    engine.output = "Considering.\n</think>\n\nBonjour."
    engine.reused = 3
    r = client.post(
        "/v1/messages",
        json={"model": "x", "max_tokens": 50, "system": "Be French.", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == [
        {"type": "thinking", "thinking": "Considering.", "signature": ""},
        {"type": "text", "text": "Bonjour."},
    ]
    assert body["stop_reason"] == "end_turn"
    assert body["usage"]["cache_read_input_tokens"] == 3
    assert body["usage"]["prompt_tokens_per_sec"] == body["usage"]["input_tokens"] * 1000
    assert tokenizer.decode(engine.prompts[0], skip_special_tokens=False).startswith("<|im_start|>system\nBe French.")


def test_anthropic_tools_both_ways(client: TestClient, engine: FakeEngine, tokenizer: Tokenizer):
    engine.output = CALL
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Oslo"}}],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "snow"}]},
    ]
    tools = [{"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]
    r = client.post(
        "/v1/messages",
        json={"max_tokens": 400, "messages": messages, "tools": tools, "thinking": {"type": "disabled"}},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["stop_reason"] == "tool_use"
    assert [(b["type"], b.get("input")) for b in body["content"]] == [("tool_use", {"city": "Paris"})]
    prompt = tokenizer.decode(engine.prompts[0], skip_special_tokens=False)
    assert "<parameter=city>\nOslo\n</parameter>" in prompt
    assert "<|im_start|>tool\nsnow" in prompt


def test_anthropic_stream(client: TestClient, engine: FakeEngine):
    engine.output = "Plan.\n</think>\n\nDone." + CALL
    r = client.post(
        "/v1/messages",
        json={
            "max_tokens": 400,
            "stream": True,
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"name": "get_weather", "input_schema": {"type": "object"}}],
        },
    )
    assert r.status_code == 200
    events = sse(r.text)
    names = [e for e, _ in events]
    assert names[0] == "message_start"
    assert names[-1] == "message_stop"
    starts = [d["content_block"]["type"] for e, d in events if e == "content_block_start"]
    assert starts == ["thinking", "text", "tool_use"]
    assert names.count("content_block_start") == names.count("content_block_stop")
    (delta,) = [d for e, d in events if e == "message_delta"]
    assert delta["delta"]["stop_reason"] == "tool_use"
    assert delta["usage"]["total_time"] == 0.003


def test_anthropic_needs_max_tokens(client: TestClient):
    assert client.post("/v1/messages", json={"messages": [{"role": "user", "content": "x"}]}).status_code == 400


def test_count_tokens(client: TestClient, tokenizer: Tokenizer):
    r = client.post("/v1/messages/count_tokens", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    expected = len(tokenizer.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n").ids)
    assert r.json() == {"input_tokens": expected}


def test_status_endpoints(client: TestClient):
    assert client.get("/v1/models").json()["data"][0]["id"] == "tiny"
    assert client.get("/health").json()["context"] == 4096
    assert "chat_template" in client.get("/props").json()
    assert client.get("/slots").json()["slots"] == 1
    chat(client)
    metrics = client.get("/metrics").json()
    assert metrics["totals"]["requests"] == 1
    assert metrics["engine"] == {"backend": "fake"}


def test_without_max_tokens_a_request_may_fill_the_context(client: TestClient, engine: FakeEngine):
    chat(client)
    assert engine.max_news == [engine.max_context - len(engine.prompts[0])]


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_the_default_limit_holds_only_without_max_tokens(model: Model, engine: FakeEngine, path: str):
    """The Anthropic API requires max_tokens, so only the OpenAI one takes the default."""
    with TestClient(create_app(model, engine, default_max_tokens=100)) as c:
        body: dict[str, object] = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 300}
        assert c.post(path, json=body).status_code == 200
        if path == "/v1/chat/completions":
            del body["max_tokens"]
            assert c.post(path, json=body).status_code == 200
            assert c.get("/props").json()["generation_settings"]["max_tokens"] == 100
    assert engine.max_news == ([300, 100] if path == "/v1/chat/completions" else [300])


def test_the_default_limit_gives_way_to_the_context(model: Model, engine: FakeEngine):
    """A prompt that leaves less room than the default writes to the end of the context, and is not refused."""
    engine.max_context = 80
    with TestClient(create_app(model, engine, default_max_tokens=100)) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "x" * 20}]})
        assert r.status_code == 200, r.text
    assert 0 < engine.max_news[0] == 80 - len(engine.prompts[0]) < 100
