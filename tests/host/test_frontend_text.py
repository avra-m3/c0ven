"""The front end's text: tokenizer, detokenizer, template and output parser, with the tiny tokenizer."""

from __future__ import annotations

import pytest
from tokenizers import Tokenizer

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
from frontend.template import ChatTemplate, Conversation, TemplateFailed
from frontend.tokenizer import Detokenizer, encode
from tests.helpers import TINY_TEMPLATE


def test_added_tokens_are_matched_whole(tokenizer: Tokenizer):
    ids = encode(tokenizer, "<|im_start|>user\n<think>")
    assert ids[0] == tokenizer.token_to_id("<|im_start|>")
    assert ids[-1] == tokenizer.token_to_id("<think>")


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


def pieces_of(out: str, thinking: bool, step: int, **kw: object) -> list[Piece]:
    p = OutputParser(thinking, **kw)  # pyright: ignore[reportArgumentType]
    got: list[Piece] = []
    for i in range(0, len(out), step):
        got += p.feed(out[i : i + step])
    return got + list(p.finish())


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


def test_string_parameters_stay_strings():
    body = "\n<function=f>\n<parameter=n>\n42\n</parameter>\n<parameter=s>\n42\n</parameter>\n</function>\n"
    call = parse_tool_call(body, {"f": {"properties": {"s": {"type": "string"}}}})
    assert call.arguments == {"n": 42, "s": "42"}
