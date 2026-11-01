"""The front end's text: tokenizer, detokenizer, template and output parser, with the tiny tokenizer."""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer

from frontend.openai_api import EFFORTS
from frontend.output import (
    Content,
    OutputParser,
    Piece,
    Reasoning,
    ToolCall,
    ToolCallError,
    ToolCallStart,
    parse_tool_call,
)
from frontend.template import LEVELS, ChatTemplate, Conversation, Efforts, TemplateFailed
from frontend.tokenizer import Detokenizer, encode, from_vocabulary
from tests.helpers import BUDGET_TEMPLATE, LEVEL_TEMPLATE, TINY_TEMPLATE, tiny_tokenizer


def test_added_tokens_are_matched_whole(tokenizer: Tokenizer):
    ids = encode(tokenizer, "<|im_start|>user\n<think>")
    assert ids[0] == tokenizer.token_to_id("<|im_start|>")
    assert ids[-1] == tokenizer.token_to_id("<think>")


def pre_tokens(tokenizer: Tokenizer, text: str) -> list[str]:
    """The pieces BPE works inside, for text NFC leaves as it is."""
    assert tokenizer.pre_tokenizer is not None
    return [text[a:b] for _, (a, b) in tokenizer.pre_tokenizer.pre_tokenize_str(text)]


def test_qwen35_splits_single_digits_after_nfc(tokenizer: Tokenizer):
    assert pre_tokens(tokenizer, "12345 中文abc") == ["1", "2", "3", "4", "5", " 中文abc"]
    assert len(encode(tokenizer, "e\u0301")) == 2, "NFC makes it one two-byte character"


def test_joyai_llm_splits_digits_in_threes_and_cjk_runs_without_normalizing():
    tk = tiny_tokenizer("joyai-llm")
    assert pre_tokens(tk, "12345 中文abc") == ["123", "45", " ", "中文", "abc"]
    assert len(encode(tk, "e\u0301")) == 3, "no NFC: e, then the two-byte accent"
    ids = encode(tk, "<|im_start|>user\n<think>")
    assert ids[0] == tk.token_to_id("<|im_start|>")
    assert ids[-1] == tk.token_to_id("<think>")


def test_an_unknown_pre_tokenizer_is_an_error():
    with pytest.raises(ValueError, match="'llama-bpe' is not one the front end builds"):
        from_vocabulary(["a"], [], [1], "llama-bpe")


def test_the_detokenizer_holds_back_a_split_character(tokenizer: Tokenizer):
    ids = encode(tokenizer, "a€b")
    assert len(ids) == 5, "the euro sign is three bytes"
    d = Detokenizer(tokenizer)
    out = [d.push([i]) for i in ids]
    assert out == ["a", "", "", "€", "b"]
    assert d.flush() == ""


def test_the_detokenizer_flushes_a_broken_tail(tokenizer: Tokenizer):
    ids = encode(tokenizer, "a€")[:-1]
    d = Detokenizer(tokenizer)
    assert "".join(d.push([i]) for i in ids) == "a"
    assert d.flush() == "�"


def test_template_settings_left_out_are_undefined():
    t = ChatTemplate(TINY_TEMPLATE)
    messages = [{"role": "user", "content": "hi"}]
    assert t.render(Conversation(messages)).endswith("<|im_start|>assistant\n<think>\n")
    assert t.render(Conversation(messages, enable_thinking=False)).endswith("<think>\n\n</think>\n\n")


def test_template_tojson_keeps_text_as_is():
    t = ChatTemplate("{% for tool in tools %}{{ tool | tojson }}{% endfor %}")
    assert t.render(Conversation([], tools=[{"d": "<café & ünïcode>"}])) == '{"d": "<café & ünïcode>"}'


def test_template_errors():
    with pytest.raises(TemplateFailed, match="no"):
        ChatTemplate("{{ raise_exception('no') }}").render(Conversation([]))


def mapping(efforts: Efforts) -> dict[str, str | None]:
    return {level: efforts.nearest(level) for level in EFFORTS if level != "none"}


def test_efforts_of_the_three_level_template_map_as_before():
    efforts = ChatTemplate(LEVEL_TEMPLATE).efforts
    assert efforts == Efforts(("low", "medium", "xhigh"), integers=False)
    assert mapping(efforts) == {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh"}


def test_efforts_of_the_budget_template():
    efforts = ChatTemplate(BUDGET_TEMPLATE).efforts
    assert efforts == Efforts(("low", "medium", "high", "max"), integers=True)
    assert mapping(efforts) == {"minimal": "low", "low": "low", "medium": "medium", "high": "high", "xhigh": "max"}


def test_a_template_that_ignores_the_effort_takes_every_level():
    efforts = ChatTemplate(TINY_TEMPLATE).efforts
    assert efforts == Efforts(LEVELS, integers=True)
    assert mapping(efforts) == {level: level for level in EFFORTS if level != "none"}


def test_a_template_that_refuses_the_probe_is_given_each_effort_as_asked():
    assert ChatTemplate("{{ raise_exception('no') }}").efforts == Efforts(LEVELS, integers=True)


def test_a_template_that_refuses_every_effort_is_given_none():
    efforts = ChatTemplate("{% if reasoning_effort is defined %}{{ raise_exception('no') }}{% endif %}").efforts
    assert efforts == Efforts((), integers=False)
    assert efforts.nearest("high") is None


def test_a_template_that_fails_on_a_level_still_loads():
    # Comparing a string level with a number fails outside Jinja's own errors.
    efforts = ChatTemplate(
        "{% if reasoning_effort is defined and reasoning_effort > 50 %}deep{% endif %}{{ messages[0].content }}"
    ).efforts
    assert efforts == Efforts((), integers=True)


def test_a_clients_own_effort_passes_as_a_string_or_an_accepted_integer():
    assert Efforts(("low",), integers=False).given("anything") == "anything"
    assert Efforts(("low",), integers=False).given(30) is None
    assert Efforts(("low",), integers=True).given(30) == 30
    assert Efforts(("low",), integers=True).given(True) is None
    assert Efforts(("low",), integers=True).given(3.5) is None


def test_preserve_thinking_sets_drop_thinking():
    t = ChatTemplate(BUDGET_TEMPLATE)
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b", "reasoning_content": "why"},
        {"role": "user", "content": "c"},
    ]
    assert "<think>why</think>" not in t.render(Conversation(messages))
    assert "<think>why</think>" not in t.render(Conversation(messages, preserve_thinking=False))
    assert "<think>why</think>" in t.render(Conversation(messages, preserve_thinking=True))


def pieces_of(out: str, thinking: bool, step: int, cut: bool = False, **kw: object) -> list[Piece]:
    p = OutputParser(thinking, **kw)  # pyright: ignore[reportArgumentType]
    got: list[Piece] = []
    for i in range(0, len(out), step):
        got += p.feed(out[i : i + step])
    return got + list(p.finish(cut))


def joined(pieces: list[Piece]) -> tuple[str, str, list[ToolCall]]:
    r = "".join(x.text for x in pieces if isinstance(x, Reasoning))
    c = "".join(x.text for x in pieces if isinstance(x, Content))
    return r, c, [x for x in pieces if isinstance(x, ToolCall)]


@pytest.mark.parametrize("step", [1, 2, 5, 1000])
def test_reasoning_content_and_calls_at_any_chunking(step: int):
    out = (
        "Check the weather.\n</think>\n\nLooking it up.\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\n"
        "Paris\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
    )
    schemas = {"get_weather": {"properties": {"city": {"type": "string"}, "days": {"type": "integer"}}}}
    r, c, calls = joined(pieces_of(out, True, step, schemas=schemas))
    assert (r, c) == ("Check the weather.", "Looking it up.")
    assert [(x.name, x.arguments) for x in calls] == [("get_weather", {"city": "Paris", "days": 3})]


@pytest.mark.parametrize("step", [1, 7, 1000])
def test_each_call_is_named_before_its_arguments(step: int):
    call = "<tool_call>\n<function={}>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>"
    pieces = pieces_of(call.format("f") + "\n" + call.format("g"), False, step)
    assert [type(x) for x in pieces] == [ToolCallStart, ToolCall, ToolCallStart, ToolCall]
    assert [x.name for x in pieces if isinstance(x, ToolCall | ToolCallStart)] == ["f", "f", "g", "g"]


def test_a_call_is_named_while_its_arguments_are_written():
    p = OutputParser(False)
    got = list(p.feed("<tool_call>\n<function=write>\n<parameter=content>\nmany lines"))
    assert got == [ToolCallStart("write")]
    assert list(p.feed(" more\n</parameter>\n</function>\n</tool_call>")) == [
        ToolCall(
            "write",
            {"content": "many lines more"},
            "\n<function=write>\n<parameter=content>\nmany lines more\n</parameter>\n</function>\n",
        )
    ]


@pytest.mark.parametrize("step", [1, 3, 1000])
def test_content_that_looks_like_a_tag(step: int):
    out = "a < b and <tool_cal is not a call. Spaces   kept."
    assert joined(pieces_of(out, False, step))[1] == out


def test_without_tools_a_tool_call_is_text():
    out = "<tool_call>x</tool_call>"
    assert joined(pieces_of(out, False, 1, tools=False))[1] == out


def test_reasoning_cut_short_by_length():
    assert joined(pieces_of("thinking about it\n", True, 1)) == ("thinking about it", "", [])


def test_an_unterminated_call_is_an_error():
    with pytest.raises(ToolCallError):
        pieces_of("<tool_call>\n<function=f>\n", False, 1000)


def test_a_call_cut_short_is_dropped():
    out = "Checking.\n<tool_call>\n<function=f>\n<parameter=x>\n4"
    for step in (1, 1000):
        assert joined(pieces_of(out, False, step, cut=True)) == ("", "Checking.", [])


def test_string_parameters_stay_strings():
    body = "\n<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=s>\n42\n</parameter>\n</function>\n"
    call = parse_tool_call(body, {"f": {"properties": {"s": {"type": "string"}}}})
    assert call.arguments == {"n": 42, "s": "42"}
