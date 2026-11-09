from __future__ import annotations

import json
from pathlib import Path

from engine.models.qwen4exp.tensors import TSV_UD_IQ4_XS, from_tsv
from engine.models.qwen4exp.turns import ChatTokens, checkpoint_rule

CHAT = ChatTokens.of(from_tsv())
checkpoint_positions = checkpoint_rule(CHAT)

USER = 846
"""The token after `<|im_start|>` in a user turn. The checkpoint rule does not look for it."""

PROMPTS = {
    row["id"]: tuple(row["tokens"])
    for row in map(json.loads, (Path(__file__).resolve().parents[3] / "reference" / "prompts.jsonl").open())
}


def test_the_generation_prompt_is_a_checkpoint():
    p = PROMPTS["chat-capital"]
    marks = checkpoint_positions(p)
    start = len(p) - 5  # <|im_start|> assistant \n <think> \n
    assert p[start : start + 2] == (CHAT.im_start, CHAT.assistant)
    assert marks == [(start, False)]


def test_every_assistant_turn_starts_a_checkpoint():
    p = PROMPTS["chat-multiturn"]
    marks = [m for m, _ in checkpoint_positions(p)]
    assert len(marks) == 2
    assert all(p[m : m + 2] == (CHAT.im_start, CHAT.assistant) for m in marks)


def test_long_prompts_every_16k():
    p = PROMPTS["doc-32k"]
    marks = checkpoint_positions(p)
    assert (16_384, False) in marks
    assert (32_768, False) not in marks, "the prompt is 32,010 tokens"


def test_a_long_system_prompt_is_kept():
    system = (CHAT.im_start, CHAT.system, CHAT.newline, *([5] * 3000), CHAT.im_end, CHAT.newline)
    p = (*system, CHAT.im_start, USER, CHAT.newline, 7, CHAT.im_end, CHAT.newline)
    assert (len(system), True) in checkpoint_positions(p)
    short = (CHAT.im_start, CHAT.system, CHAT.newline, 5, CHAT.im_end, CHAT.newline, 9)
    assert checkpoint_positions(short) == []


def test_the_ids_are_the_files():
    """The ids the rule looks for, in both committed tables: the ones spec/MODEL_IO.md, section 4.1 gives."""
    assert CHAT == ChatTokens.of(from_tsv(TSV_UD_IQ4_XS)) == ChatTokens(248_045, 248_046, 8_678, 74_455, 198)
