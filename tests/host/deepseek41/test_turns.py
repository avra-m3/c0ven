from __future__ import annotations

from engine.models.deepseek41 import numbers
from engine.models.deepseek41.turns import LONG_SYSTEM, checkpoint_positions

# The file's tokenizer ids, written out so that a wrong id in numbers.py fails here.
BOS = 0
EOS = 1
SYSTEM = 128_799
USER = 128_803
ASSISTANT = 128_804
WORD = 5_000


def test_the_token_ids_are_the_tokenizers():
    assert (numbers.EOS, numbers.USER, numbers.ASSISTANT) == (EOS, USER, ASSISTANT)
    assert numbers.EOS_IDS == (EOS,)


def test_every_assistant_turn_starts_a_checkpoint():
    p = (BOS, USER, WORD, ASSISTANT, WORD, EOS, USER, WORD, ASSISTANT, WORD)
    assert checkpoint_positions(p) == [(3, False), (8, False)]


def test_a_long_system_message_is_kept_where_it_ends():
    system = (BOS, SYSTEM, *(WORD,) * LONG_SYSTEM)
    p = (*system, USER, WORD, ASSISTANT)
    assert checkpoint_positions(p) == [(len(system), True), (len(system) + 2, False)]


def test_only_the_first_user_turn_ends_the_system_message():
    system = (BOS, SYSTEM, *(WORD,) * LONG_SYSTEM)
    p = (*system, USER, WORD, ASSISTANT, WORD, EOS, USER, WORD, ASSISTANT)
    n = len(system)
    assert checkpoint_positions(p) == [(n, True), (n + 2, False), (n + 7, False)]


def test_a_short_system_message_is_not():
    p = (BOS, SYSTEM, WORD, USER, WORD, ASSISTANT)
    assert checkpoint_positions(p) == [(5, False)]


def test_a_late_user_turn_after_a_short_system_message_is_not_kept():
    p = (BOS, SYSTEM, WORD, USER, *(WORD,) * LONG_SYSTEM, EOS, USER, WORD, ASSISTANT)
    assert checkpoint_positions(p) == [(len(p) - 1, False)]


def test_long_prompts_every_16k():
    p = (BOS, USER, *(WORD,) * 40_000)
    assert checkpoint_positions(p) == [(16_384, False), (32_768, False)]
