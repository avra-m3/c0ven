"""Splitting the model's output into reasoning, content and tool calls, as it streams.

The generation prompt ends in `<think>`, with or without a newline, when thinking is on, so the output starts as
reasoning and turns to content at `</think>`. Tool calls come in blocks, in the format the chat template teaches
(XML_FORMAT or DSML_FORMAT).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"

_FUNCTION_START = re.compile(r"<function=([^>\n]+)>")
_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)

DSML = "\uff5cDSML\uff5c"
"""The token that opens DeepSeek's tool-call tags, between fullwidth bars."""
_INVOKE_START = re.compile(f'<{DSML} invoke name="([^"\\n]*)">')
_INVOKE = re.compile(f'<{DSML} invoke name="([^"\\n]*)">(.*)', re.DOTALL)
_DSML_PARAMETER = re.compile(
    f'<{DSML} parameter name="([^"\\n]*)" string="(true|false)">(.*?)</{DSML} parameter>', re.DOTALL
)


@dataclass(frozen=True, slots=True)
class Reasoning:
    text: str


@dataclass(frozen=True, slots=True)
class Content:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallStart:
    """A tool call has begun; its ToolCall follows, unless the output ends first."""

    name: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


type Piece = Reasoning | Content | ToolCallStart | ToolCall


class ToolCallError(ValueError):
    pass


def _strip_one_newline(text: str) -> str:
    text = text.removeprefix("\n")
    return text.removesuffix("\n")


def parse_tool_call(body: str, schemas: Mapping[str, Mapping[str, Any]] | None = None) -> ToolCall:
    m = _FUNCTION.search(body)
    if m is None:
        raise ToolCallError(f"no <function=...> block in {body[:80]!r}")
    name = m.group(1).strip()
    properties: Mapping[str, Any] = {}
    if schemas is not None and name in schemas:
        properties = schemas[name].get("properties", {}) or {}
    arguments: dict[str, Any] = {}
    for p in _PARAMETER.finditer(m.group(2)):
        key = p.group(1).strip()
        value = _strip_one_newline(p.group(2))
        kind = properties.get(key, {}).get("type") if isinstance(properties.get(key), Mapping) else None
        if kind == "string":
            arguments[key] = value
            continue
        try:
            arguments[key] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value
    return ToolCall(name, arguments, body)


def parse_dsml_call(body: str, schemas: Mapping[str, Mapping[str, Any]] | None = None) -> ToolCall:
    """One `<|DSML| invoke>` call, up to its closing tag. The `string` attribute gives each value's type, so the
    schemas are not read."""
    m = _INVOKE.search(body)
    if m is None:
        raise ToolCallError(f"no <{DSML} invoke name=...> tag in {body[:80]!r}")
    arguments: dict[str, Any] = {}
    for p in _DSML_PARAMETER.finditer(m.group(2)):
        key, string, value = p.groups()
        if string == "true":
            arguments[key] = value
            continue
        try:
            arguments[key] = json.loads(value)
        except json.JSONDecodeError:
            arguments[key] = value
    return ToolCall(m.group(1).strip(), arguments, body)


type CallParser = Callable[[str, Mapping[str, Mapping[str, Any]] | None], ToolCall]


@dataclass(frozen=True, slots=True)
class ToolCallFormat:
    """How a chat template writes tool calls."""

    start: str
    """Opens a block of calls."""
    end: str
    """Closes the block."""
    name: re.Pattern[str]
    """Matches the start of a call, with its name in group 1. A match never holds a newline."""
    parse: CallParser
    """Reads one call's text: the block's, or with `call_end`, from the last call's end to this one's."""
    call_end: str | None = None
    """Closes each call of a block that holds several; None for one call a block."""


XML_FORMAT = ToolCallFormat(CALL_START, CALL_END, _FUNCTION_START, parse_tool_call)
"""Qwen's format, one call a block (spec/MODEL_IO.md, section 4.2). A value is kept as text when the tool's schema
says the parameter is a string, and read as JSON otherwise, falling back to text:

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>
"""

DSML_FORMAT = ToolCallFormat(f"<{DSML} calls>", f"</{DSML} calls>", _INVOKE_START, parse_dsml_call, f"</{DSML} invoke>")
"""DeepSeek V4.1's format, several calls a block (spec/MODEL_IO.md, section 4.2). The bars around DSML are fullwidth
(U+FF5C). `string="true"` marks a value as text, and `string="false"` as JSON, falling back to text:

    <|DSML| calls>
    <|DSML| invoke name="NAME">
    <|DSML| parameter name="KEY" string="true">VALUE</|DSML| parameter>
    </|DSML| invoke>
    </|DSML| calls>
"""


def call_format(template: str) -> ToolCallFormat:
    """The format of the calls a chat template's source teaches: DSML when it holds the DSML token, else XML."""
    return DSML_FORMAT if DSML in template else XML_FORMAT


class _Mode(Enum):
    REASONING = auto()
    CONTENT = auto()
    TOOL_CALL = auto()


def _held(text: str, tag: str) -> int:
    """How many characters at the end of `text` could be the start of `tag`."""
    for n in range(min(len(tag) - 1, len(text)), 0, -1):
        if tag.startswith(text[-n:]):
            return n
    return 0


def _trailing_space(text: str) -> int:
    return len(text) - len(text.rstrip())


@dataclass(slots=True)
class OutputParser:
    """Feed it text as it is detokenized; it yields the pieces that are certain. Whitespace before `</think>` and a
    tool-call block, and after `</think>`, is dropped, so it is held back until what follows is known. A call's name
    is yielded as soon as it is read, and its arguments once the call closes."""

    thinking: bool
    """Whether the prompt ended inside `<think>`."""
    schemas: Mapping[str, Mapping[str, Any]] | None = None
    tools: bool = True
    call_format: ToolCallFormat = XML_FORMAT
    calls: list[ToolCall] = field(default_factory=lambda: [])
    _mode: _Mode = _Mode.CONTENT
    _buffer: str = ""
    _started: bool = False
    """Something has been yielded in the current mode."""
    _named: bool = False
    """The current tool call's ToolCallStart has been yielded."""
    _scanned: int = 0
    """The buffer's first `_scanned` characters hold no whole mark that the current mode looks for."""
    _name_from: int = 0
    """No call name starts before this offset in the buffer."""

    def __post_init__(self) -> None:
        self._mode = _Mode.REASONING if self.thinking else _Mode.CONTENT

    def feed(self, text: str) -> Iterator[Piece]:
        self._buffer += text
        while True:
            match self._mode:
                case _Mode.REASONING:
                    if not (yield from self._until(THINK_END, Reasoning)):
                        return
                    self._switch(_Mode.CONTENT)
                case _Mode.CONTENT:
                    if not self.tools:
                        yield from self._until(None, Content)
                        return
                    if not (yield from self._until(self.call_format.start, Content)):
                        return
                    self._switch(_Mode.TOOL_CALL)
                case _Mode.TOOL_CALL:
                    fmt = self.call_format
                    end = self._find(fmt.end)
                    i = self._find(fmt.call_end) if fmt.call_end else -1
                    # In a block of several calls, a name after this call's end or the block's belongs to a later one.
                    marks = (end, i) if fmt.call_end else ()
                    stop = min((x for x in marks if x >= 0), default=len(self._buffer))
                    if not self._named and (m := fmt.name.search(self._buffer, self._name_from, stop)):
                        self._named = True
                        yield ToolCallStart(m.group(1).strip())
                    if fmt.call_end and i >= 0 and (end < 0 or i < end):
                        yield from self._call(self._buffer[:i])
                        self._drop(i + len(fmt.call_end))
                        self._named = False
                        continue
                    if end < 0:
                        self._scanned = len(self._buffer)
                        if not self._named:
                            # A name yet to come lies after the last newline, as no name spans one.
                            self._name_from = max(self._name_from, self._buffer.rfind("\n", self._name_from) + 1)
                        return
                    if fmt.call_end is None:
                        yield from self._call(self._buffer[:end])
                    elif self._named:
                        # Its ToolCallStart is out, and a client expects the arguments to follow.
                        raise ToolCallError(f"the tool-call block closed inside a call: {self._buffer[:80]!r}")
                    self._drop(end + len(fmt.end))
                    self._switch(_Mode.CONTENT)

    def finish(self, cut: bool = False) -> Iterator[Piece]:
        """The end of the output: whatever is held back. An unterminated tool call is dropped when the output was cut
        (`cut`: a token limit or a cancel), and is an error when the model ended it."""
        rest = self._buffer.rstrip()
        self._drop(len(self._buffer))
        match self._mode:
            case _Mode.REASONING:
                yield from self._piece(Reasoning, rest)
            case _Mode.CONTENT:
                yield from self._piece(Content, rest)
            case _Mode.TOOL_CALL:
                if not cut:
                    raise ToolCallError(f"the output ended inside a tool call: {rest[:80]!r}")

    def _call(self, text: str) -> Iterator[Piece]:
        call = self.call_format.parse(text, self.schemas)
        if not self._named:
            yield ToolCallStart(call.name)
        self.calls.append(call)
        yield call

    def _find(self, mark: str) -> int:
        """Where `mark` first is in the buffer. The search starts `len(mark) - 1` characters before the part not yet
        scanned, where a mark can straddle the boundary, so it covers each piece's new text and not the whole reply."""
        return self._buffer.find(mark, max(0, self._scanned - len(mark) + 1))

    def _drop(self, n: int) -> None:
        """Drop the buffer's first `n` characters; what is left has not been scanned."""
        self._buffer = self._buffer[n:]
        self._scanned = 0
        self._name_from = 0

    def _switch(self, mode: _Mode) -> None:
        self._mode = mode
        self._started = False
        self._named = False

    def _until(self, tag: str | None, kind: type[Reasoning] | type[Content]) -> Iterator[Piece]:
        """Yield the buffer's text up to `tag`, consuming the tag. Returns whether the tag was found."""
        i = self._find(tag) if tag else -1
        if i >= 0:
            yield from self._piece(kind, self._buffer[:i].rstrip())
            self._drop(i + len(tag or ""))
            return True
        held = _held(self._buffer, tag) if tag else 0
        safe = len(self._buffer) - held
        safe -= _trailing_space(self._buffer[:safe])
        yield from self._piece(kind, self._buffer[:safe])
        self._buffer = self._buffer[safe:]
        self._scanned = len(self._buffer)
        return False

    def _piece(self, kind: type[Reasoning] | type[Content], text: str) -> Iterator[Piece]:
        if not self._started:
            text = text.lstrip()
        if text:
            self._started = True
            yield kind(text)
