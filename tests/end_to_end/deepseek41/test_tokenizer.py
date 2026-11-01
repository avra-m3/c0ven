"""The tokenizer the front end builds from DeepSeek-V4.1-Flash's GGUF, against DeepSeek's own.

The model is read from `DEEPSEEK41_MODEL_DIR`: its first GGUF shard gives the vocabulary, the merges and the token
types, as the front end reads them in service. The expected ids are what DeepSeek's tokenizer.json (Hugging Face
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at 1bae0e1) gives, with no special tokens added.
"""

# DeepSeek's markers are written with the full-width bar, U+FF5C.
# ruff: noqa: RUF001

from __future__ import annotations

import os
from pathlib import Path

import pytest

from frontend.model import Model, load
from frontend.tokenizer import encode

pytestmark = pytest.mark.model

EXPECTED = {
    "english": (
        "The quick brown fox doesn't jump; it's 3x faster.\n",
        [671, 4787, 13769, 46012, 5354, 1664, 12583, 29, 436, 734, 223, 21, 90, 13058, 603],
    ),
    "code": (
        "def f(x):\n    return [i**2 for i in range(x)]\n",
        [3465, 285, 4042, 3395, 361, 1354, 764, 75, 666, 20, 362, 1008, 295, 3291, 4042, 22143],
    ),
    "numbers": (
        "1234567 3.14159 2026-11-01 $1,234.56",
        [6895, 18009, 25, 223, 21, 16, 9926, 3318, 223, 939, 24, 15, 779, 15, 2616, 957, 19, 14, 14456, 16, 3661],
    ),
    "cjk": (
        "我爱自然语言处理。東京タワー、ひらがな",
        [60147, 4377, 7831, 4398, 320, 66771, 11767, 37560, 4045, 410, 40259, 4970, 2936, 2942],
    ),
    "chat": (
        "<｜begin▁of▁sentence｜>You are helpful.<｜User｜>Hi<｜Assistant｜><think>Greet.</think>Hello!"
        "<｜end▁of▁sentence｜>",
        [0, 3476, 477, 11502, 16, 128803, 23166, 128804, 128821, 41467, 319, 16, 128822, 19923, 3, 1],
    ),
    "dsml": (
        '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="f">',
        [30, 128825, 72461, 4941, 12548, 1018, 30, 128825, 40148, 5406, 2329, 1281, 72, 3320],
    ),
}


@pytest.fixture(scope="module")
def model() -> Model:
    root = os.environ.get("DEEPSEEK41_MODEL_DIR")
    if root is None:
        pytest.skip("set DEEPSEEK41_MODEL_DIR to the directory of DeepSeek-V4.1-Flash's GGUF shards")
    shards = sorted(Path(root).glob("*-00001-of-*.gguf")) or sorted(Path(root).glob("*.gguf"))
    if not shards:
        pytest.skip(f"no GGUF file in {root}")
    m = load(shards[0])
    if m.arch != "deepseek41":
        pytest.skip(f"{shards[0]} is {m.arch}, not deepseek41")
    return m


@pytest.mark.parametrize("name", EXPECTED)
def test_text_tokenizes_as_deepseeks_tokenizer_does(model: Model, name: str):
    text, ids = EXPECTED[name]
    assert encode(model.tokenizer, text) == ids


MARKERS = {"<think>": 128821, "</think>": 128822, "<｜User｜>": 128803, "<｜Assistant｜>": 128804, "｜DSML｜": 128825}


@pytest.mark.parametrize("token", MARKERS)
def test_the_templates_markers_are_one_token(model: Model, token: str):
    assert encode(model.tokenizer, token) == [MARKERS[token]]
