from __future__ import annotations

import json
from pathlib import Path

from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.turns import checkpoint_positions

PROMPTS = {
    row["id"]: tuple(row["tokens"])
    for row in map(json.loads, (Path(__file__).resolve().parents[3] / "reference" / "prompts.jsonl").open())
}


def test_the_generation_prompt_is_a_checkpoint():
    p = PROMPTS["chat-capital"]
    marks = checkpoint_positions(p)
    start = len(p) - 5  # <|im_start|> assistant \n <think> \n
    assert p[start : start + 2] == (numbers.IM_START, numbers.ASSISTANT)
    assert marks == [(start, False)]


def test_every_assistant_turn_starts_a_checkpoint():
    p = PROMPTS["chat-multiturn"]
    marks = [m for m, _ in checkpoint_positions(p)]
    assert len(marks) == 2
    assert all(p[m : m + 2] == (numbers.IM_START, numbers.ASSISTANT) for m in marks)


def test_long_prompts_every_16k():
    p = PROMPTS["doc-32k"]
    marks = checkpoint_positions(p)
    assert (16_384, False) in marks
    assert (32_768, False) not in marks, "the prompt is 32,010 tokens"


def test_a_long_system_prompt_is_kept():
    system = (numbers.IM_START, numbers.SYSTEM, numbers.NEWLINE, *([5] * 3000), numbers.IM_END, numbers.NEWLINE)
    p = (*system, numbers.IM_START, numbers.USER, numbers.NEWLINE, 7, numbers.IM_END, numbers.NEWLINE)
    assert (len(system), True) in checkpoint_positions(p)
    short = (numbers.IM_START, numbers.SYSTEM, numbers.NEWLINE, 5, numbers.IM_END, numbers.NEWLINE, 9)
    assert checkpoint_positions(short) == []
