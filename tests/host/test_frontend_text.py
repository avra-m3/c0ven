"""The front end's text: tokenizer, detokenizer, template and output parser, with the tiny tokenizer."""

from __future__ import annotations

from operator import index
from typing import ClassVar, SupportsIndex

import pytest
from tokenizers import Tokenizer

from frontend.openai_api import EFFORTS
from frontend.output import (
    DSML,
    DSML_FORMAT,
    THINK_END,
    XML_FORMAT,
    Content,
    OutputParser,
    Piece,
    Reasoning,
    ToolCall,
    ToolCallError,
    ToolCallFormat,
    ToolCallStart,
    parse_dsml_call,
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


def dsml(text: str) -> str:
    """DeepSeek's DSML token has fullwidth bars (U+FF5C); the samples spell them `|`."""
    return text.replace("|DSML|", DSML)


# Completions as DeepSeek's encoding writes them (deepseek-ai/DeepSeek-V4.1-Flash-1017, encoding/test_encoding.py
# and encoding/tests/), without the end-of-sentence token, which the engine stops at and does not send.
LOOKUP = dsml(
    "  reason  </think>summary\n\n"
    "<|DSML| calls>\n"
    '<|DSML| invoke name="lookup">\n'
    '<|DSML| parameter name="query" string="true">value</|DSML| parameter>\n'
    '<|DSML| parameter name="limit" string="false">2</|DSML| parameter>\n'
    "</|DSML| invoke>\n"
    "</|DSML| calls>"
)
NAMESPACES = dsml(
    "summary\n\n<|DSML| calls>\n"
    + "\n".join(
        f'<|DSML| invoke name="{name}">\n'
        '<|DSML| parameter name="query" string="true">value</|DSML| parameter>\n'
        "</|DSML| invoke>"
        for name in ("lookup", "search::lookup", "files::lookup")
    )
    + "\n</|DSML| calls>"
)
WEATHER = dsml(
    "The user wants the weather in Beijing. I should call get_weather.</think>\n\n"
    "<|DSML| calls>\n"
    '<|DSML| invoke name="get_weather">\n'
    '<|DSML| parameter name="location" string="true">Beijing</|DSML| parameter>\n'
    '<|DSML| parameter name="unit" string="true">celsius</|DSML| parameter>\n'
    "</|DSML| invoke>\n"
    "</|DSML| calls>"
)
SEARCH = dsml(
    "\n\n<|DSML| calls>\n"
    '<|DSML| invoke name="search">\n'
    '<|DSML| parameter name="queries" string="true">小柴胡冲剂 布洛芬 相互作用 一起吃</|DSML| parameter>\n'
    "</|DSML| invoke>\n"
    "</|DSML| calls>"
)
# The `open` tool of encoding/tests/test_input_3.json, its arguments written as encode_arguments_to_dsml writes them.
OPEN = dsml(
    "Opening both.\n\n<|DSML| calls>\n"
    '<|DSML| invoke name="open">\n'
    '<|DSML| parameter name="open_list" string="false">[{"id": 0, "loc": 12, "num_lines": 40}, '
    '{"id": "https://example.com/a b"}]</|DSML| parameter>\n'
    '<|DSML| parameter name="note" string="true">{"not": "json"}\n second line</|DSML| parameter>\n'
    '<|DSML| parameter name="strict" string="false">true</|DSML| parameter>\n'
    "</|DSML| invoke>\n"
    '<|DSML| invoke name="search">\n'
    '<|DSML| parameter name="queries" string="true">42</|DSML| parameter>\n'
    "</|DSML| invoke>\n"
    "</|DSML| calls>"
)
STEPS = [1, 2, 3, 5, 7, 11, 1000]


def dsml_pieces(out: str, thinking: bool, step: int) -> list[Piece]:
    return pieces_of(out, thinking, step, call_format=DSML_FORMAT)


@pytest.mark.parametrize("step", STEPS)
def test_dsml_reasoning_content_and_a_call(step: int):
    r, c, calls = joined(dsml_pieces(LOOKUP, True, step))
    assert (r, c) == ("reason", "summary")
    assert [(x.name, x.arguments) for x in calls] == [("lookup", {"query": "value", "limit": 2})]


@pytest.mark.parametrize("step", STEPS)
def test_dsml_several_calls_are_each_named_before_their_arguments(step: int):
    pieces = dsml_pieces(NAMESPACES, False, step)
    assert joined(pieces)[1] == "summary"
    calls = [x for x in pieces if not isinstance(x, Content)]
    assert [type(x) for x in calls] == [ToolCallStart, ToolCall] * 3
    names = ["lookup", "lookup", "search::lookup", "search::lookup", "files::lookup", "files::lookup"]
    assert [x.name for x in calls if isinstance(x, ToolCall | ToolCallStart)] == names
    assert all(x.arguments == {"query": "value"} for x in calls if isinstance(x, ToolCall))


@pytest.mark.parametrize("step", STEPS)
def test_dsml_calls_after_reasoning_alone(step: int):
    r, c, calls = joined(dsml_pieces(WEATHER, True, step))
    assert (r, c) == ("The user wants the weather in Beijing. I should call get_weather.", "")
    assert [(x.name, x.arguments) for x in calls] == [("get_weather", {"location": "Beijing", "unit": "celsius"})]
    r, c, calls = joined(dsml_pieces(SEARCH, False, step))
    assert (r, c) == ("", "")
    assert [(x.name, x.arguments) for x in calls] == [("search", {"queries": "小柴胡冲剂 布洛芬 相互作用 一起吃"})]


@pytest.mark.parametrize("step", STEPS)
def test_dsml_values_are_json_or_text_as_marked(step: int):
    _, c, calls = joined(dsml_pieces(OPEN, False, step))
    assert c == "Opening both."
    assert [(x.name, x.arguments) for x in calls] == [
        (
            "open",
            {
                "open_list": [{"id": 0, "loc": 12, "num_lines": 40}, {"id": "https://example.com/a b"}],
                "note": '{"not": "json"}\n second line',
                "strict": True,
            },
        ),
        ("search", {"queries": "42"}),
    ]


@pytest.mark.parametrize("step", STEPS)
def test_dsml_a_later_block_does_not_name_a_call_early(step: int):
    block = WEATHER.split("</think>")[1]
    pieces = dsml_pieces(block + "\n\nAgain." + block + dsml('\n\nThen <|DSML| invoke name="x"> as text.'), False, step)
    assert [type(x) for x in pieces if not isinstance(x, Content)] == [ToolCallStart, ToolCall] * 2
    assert [x.name for x in pieces if isinstance(x, ToolCall)] == ["get_weather", "get_weather"]
    assert joined(pieces)[1] == "Again." + dsml('Then <|DSML| invoke name="x"> as text.')


def test_dsml_marks_split_anywhere_are_held_back():
    for cut in range(len("summary\n\n"), len(LOOKUP.split("</think>")[1])):
        p = OutputParser(False, call_format=DSML_FORMAT)
        text = LOOKUP.split("</think>")[1]
        got = list(p.feed(text[:cut])) + list(p.feed(text[cut:])) + list(p.finish())
        assert got[0] == Content("summary"), cut
        assert [type(x) for x in got[1:]] == [ToolCallStart, ToolCall], cut


def test_dsml_a_call_is_named_while_its_arguments_are_written():
    p = OutputParser(False, call_format=DSML_FORMAT)
    head = dsml('Writing.\n\n<|DSML| calls>\n<|DSML| invoke name="write">\n<|DSML| parameter name="content" string="t')
    assert list(p.feed(head)) == [Content("Writing."), ToolCallStart("write")]
    assert list(p.feed(dsml('rue">many lines'))) == []
    tail = dsml(" more</|DSML| parameter>\n</|DSML| invoke>\n</|DSML| calls>")
    assert [(x.name, x.arguments) for x in p.feed(tail) if isinstance(x, ToolCall)] == [
        ("write", {"content": "many lines more"})
    ]
    assert list(p.finish()) == []


def test_dsml_content_that_looks_like_a_mark():
    out = dsml("Use <|DSML| call or <tool_call> as text.")
    assert joined(pieces_of(out, False, 1, call_format=DSML_FORMAT))[1] == out


def test_dsml_errors():
    with pytest.raises(ToolCallError):
        dsml_pieces(dsml('<|DSML| calls>\n<|DSML| invoke name="f">\n'), False, 1000)
    with pytest.raises(ToolCallError):
        dsml_pieces(dsml('<|DSML| calls>\n<|DSML| invoke name="f">\n</|DSML| calls>'), False, 1000)
    with pytest.raises(ToolCallError):
        parse_dsml_call("no invoke here")


def test_the_template_chooses_the_call_format():
    assert ChatTemplate(TINY_TEMPLATE).tool_calls is XML_FORMAT
    assert ChatTemplate(dsml("{{- '<|DSML| calls>' }}")).tool_calls is DSML_FORMAT


def long_reply(fmt: ToolCallFormat) -> str:
    """Reasoning, content and calls long enough that rescanning the whole reply on each piece would show."""
    words = " ".join(f"w{i} < >" for i in range(2000))
    if fmt is XML_FORMAT:
        call = "<tool_call>\n<function=f{}>\n<parameter=text>\n{}\n</parameter>\n</function>\n</tool_call>"
        calls = "\n".join(call.format(n, words) for n in range(3))
    else:
        call = dsml('<|DSML| invoke name="f{}">\n<|DSML| parameter name="text" string="true">{}</|DSML| parameter>\n')
        calls = dsml("<|DSML| calls>\n") + "".join(call.format(n, words) + dsml("</|DSML| invoke>\n") for n in range(3))
        calls += dsml("</|DSML| calls>")
    return f"{words}\n</think>\n\n{words}\n\n{calls}"


def outcome(pieces: list[Piece]) -> tuple[str, str, list[tuple[str, str]]]:
    r, c, _ = joined(pieces)
    calls = [(type(x).__name__, repr(x)) for x in pieces if isinstance(x, ToolCall | ToolCallStart)]
    return r, c, calls


@pytest.mark.parametrize("fmt", [XML_FORMAT, DSML_FORMAT])
def test_a_long_reply_fed_a_character_at_a_time_reads_as_fed_whole(fmt: ToolCallFormat):
    out = long_reply(fmt)
    whole = outcome(pieces_of(out, True, len(out), call_format=fmt))
    assert whole[0]
    assert whole[1]
    assert len(whole[2]) == 6
    assert outcome(pieces_of(out, True, 1, call_format=fmt)) == whole


@pytest.mark.parametrize("fmt", [XML_FORMAT, DSML_FORMAT])
def test_marks_split_across_pieces(fmt: ToolCallFormat):
    out = long_reply(fmt)
    whole = outcome(pieces_of(out, True, len(out), call_format=fmt))
    name = fmt.name.search(out)
    assert name
    marks = [THINK_END, fmt.start, fmt.end, *([fmt.call_end] if fmt.call_end else [])]
    spans = [(out.index(m), out.index(m) + len(m)) for m in marks]
    # Each mark is cut after each of its characters, once the parser has scanned a long first piece.
    for start, end in [*spans, name.span()]:
        for cut in range(start + 1, end):
            p = OutputParser(True, call_format=fmt)
            got = list(p.feed(out[: start - 7])) + list(p.feed(out[start - 7 : cut])) + list(p.feed(out[cut:]))
            assert outcome(got + list(p.finish())) == whole, (start, cut)


class SearchedText(str):
    """A parser's buffer that records how many characters each find and rfind is given, and stays a SearchedText
    when added to or sliced."""

    widths: ClassVar[list[int]] = []

    def find(self, sub: str, start: SupportsIndex | None = None, end: SupportsIndex | None = None, /) -> int:
        SearchedText.widths.append(len(self) - index(start or 0))
        return super().find(sub, start, end)

    def rfind(self, sub: str, start: SupportsIndex | None = None, end: SupportsIndex | None = None, /) -> int:
        SearchedText.widths.append(len(self) - index(start or 0))
        return super().rfind(sub, start, end)

    def __add__(self, other: str) -> SearchedText:
        return SearchedText(str.__add__(self, other))

    def __getitem__(self, key: SupportsIndex | slice) -> SearchedText:
        return SearchedText(super().__getitem__(key))


@pytest.mark.parametrize("fmt", [XML_FORMAT, DSML_FORMAT])
def test_each_piece_is_searched_only_near_its_new_text(fmt: ToolCallFormat):
    out = long_reply(fmt)
    p = OutputParser(True, call_format=fmt)
    p._buffer = SearchedText()  # pyright: ignore[reportPrivateUsage]
    SearchedText.widths = []
    for ch in out:
        list(p.feed(ch))
    assert len(SearchedText.widths) > len(out)
    # A search reaches back no further than a mark or a call's name can straddle.
    assert max(SearchedText.widths) < 64


@pytest.mark.parametrize("step", STEPS)
def test_dsml_a_call_without_a_name_is_not_named_from_the_next(step: int):
    out = dsml('<|DSML| calls>\n<|DSML| invoke>\n</|DSML| invoke>\n<|DSML| invoke name="f">\n</|DSML| invoke>\n')
    p = OutputParser(False, call_format=DSML_FORMAT)
    got: list[Piece] = []

    def feed() -> None:
        for i in range(0, len(out), step):
            got.extend(p.feed(out[i : i + step]))

    with pytest.raises(ToolCallError):
        feed()
    assert got == []
