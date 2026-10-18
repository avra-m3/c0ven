"""The front end's text against the real model's tokenizer and chat template (TESTING.md, level 7's text half).

The model is read from `FLASH_MODEL_DIR` (default `/models`): its first GGUF shard gives the vocabulary, the merges
and the template, as the front end reads them in service. The expected text is in `reference/` (REFERENCE_OUTPUTS.md).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from frontend.model import Model, load
from frontend.template import Conversation
from frontend.tokenizer import Detokenizer, encode

pytestmark = pytest.mark.model

ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ROOT / "reference" / "prompts.jsonl"
GREEDY = ROOT / "reference" / "llamacpp-3cf0325-IQ3_S-greedy.jsonl"


def rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [r for line in f if "meta" not in (r := json.loads(line))]


@pytest.fixture(scope="module")
def model() -> Model:
    root = Path(os.environ.get("FLASH_MODEL_DIR", "/models"))
    shards = sorted(root.glob("*-00001-of-*.gguf")) or sorted(root.glob("*.gguf"))
    if not shards:
        pytest.skip(f"no GGUF file in {root} (set FLASH_MODEL_DIR)")
    return load(shards[0])


PROMPT_ROWS = rows(PROMPTS)


@pytest.mark.parametrize("row", PROMPT_ROWS, ids=[r["id"] for r in PROMPT_ROWS])
def test_reference_prompts_tokenize_as_llama_cpp_did(model: Model, row: dict[str, Any]):
    assert encode(model.tokenizer, row["text"]) == row["tokens"]


def test_streamed_text_is_the_whole_decode(model: Model):
    for row in rows(GREEDY):
        ids = [i for i in row["tokens"] if i not in model.stop_ids]
        d = Detokenizer(model.tokenizer)
        streamed = "".join(d.push([i]) for i in ids) + d.flush()
        assert streamed == model.tokenizer.decode(ids, skip_special_tokens=False), row["id"]


def by_id(name: str) -> str:
    (row,) = [r for r in PROMPT_ROWS if r["id"] == name]
    return row["text"]


def test_template_renders_a_system_prompt(model: Model):
    messages = [
        {
            "role": "system",
            "content": "You are a pirate. Answer every question in the voice of a pirate, in at most three sentences.",
        },
        {"role": "user", "content": "How do I boil an egg so the yolk stays runny?"},
    ]
    assert model.template.render(Conversation(messages, enable_thinking=False)) == by_id("chat-system")


def test_template_renders_earlier_turns(model: Model):
    messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "I'm planning a three-day trip to Lisbon in March. What should I pack?"},
        {
            "role": "assistant",
            "content": "March in Lisbon is mild but changeable: pack layers, a light waterproof jacket, comfortable "
            "walking shoes for the hills and cobblestones, and a small umbrella.",
        },
        {"role": "user", "content": "Good. And is it worth getting the Lisbon Card?"},
    ]
    assert model.template.render(Conversation(messages, enable_thinking=False)) == by_id("chat-multiturn")
