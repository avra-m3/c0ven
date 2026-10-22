"""Splitting the model's output into reasoning, content and tool calls, as it streams.

The generation prompt opens `<think>\\n` when thinking is on, so the output starts as reasoning and turns to content
at `</think>`. Tool calls are the template's format (spec/MODEL_IO.md, section 4.2):

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

A value is kept as text when the tool's schema says the parameter is a string, and read as JSON otherwise, falling
back to text.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"

_FUNCTION = re.compile(r"<function=([^>\n]+)>(.*?)</function>", re.DOTALL)
_PARAMETER = re.compile(r"<parameter=([^>\n]+)>(.*?)</parameter>", re.DOTALL)


@dataclass(frozen=True, slots=True)
class Reasoning:
    text: str


@dataclass(frozen=True, slots=True)
class Content:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""


type Piece = Reasoning | Content | ToolCall


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
    """Feed it text as it is detokenized; it yields the pieces that are certain. Whitespace before `</think>` and
    `<tool_call>`, and after `</think>`, is dropped, so it is held back until what follows is known."""

    thinking: bool
    """Whether the prompt ended inside `<think>`."""
    schemas: Mapping[str, Mapping[str, Any]] | None = None
    tools: bool = True
    calls: list[ToolCall] = field(default_factory=lambda: [])
    _mode: _Mode = _Mode.CONTENT
    _buffer: str = ""
    _started: bool = False
    """Something has been yielded in the current mode."""

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
                    if not (yield from self._until(CALL_START, Content)):
                        return
                    self._switch(_Mode.TOOL_CALL)
                case _Mode.TOOL_CALL:
                    i = self._buffer.find(CALL_END)
                    if i < 0:
                        return
                    call = parse_tool_call(self._buffer[:i], self.schemas)
                    self.calls.append(call)
                    yield call
                    self._buffer = self._buffer[i + len(CALL_END) :]
                    self._switch(_Mode.CONTENT)

    def finish(self) -> Iterator[Piece]:
        """The end of the output: whatever is held back. An unterminated tool call is an error."""
        rest, self._buffer = self._buffer.rstrip(), ""
        match self._mode:
            case _Mode.REASONING:
                yield from self._piece(Reasoning, rest)
            case _Mode.CONTENT:
                yield from self._piece(Content, rest)
            case _Mode.TOOL_CALL:
                raise ToolCallError(f"the output ended inside a tool call: {rest[:80]!r}")

    def _switch(self, mode: _Mode) -> None:
        self._mode = mode
        self._started = False

    def _until(self, tag: str | None, kind: type[Reasoning] | type[Content]) -> Iterator[Piece]:
        """Yield the buffer's text up to `tag`, consuming the tag. Returns whether the tag was found."""
        i = self._buffer.find(tag) if tag else -1
        if i >= 0:
            yield from self._piece(kind, self._buffer[:i].rstrip())
            self._buffer = self._buffer[i + len(tag or "") :]
            return True
        held = _held(self._buffer, tag) if tag else 0
        safe = len(self._buffer) - held
        safe -= _trailing_space(self._buffer[:safe])
        yield from self._piece(kind, self._buffer[:safe])
        self._buffer = self._buffer[safe:]
        return False

    def _piece(self, kind: type[Reasoning] | type[Content], text: str) -> Iterator[Piece]:
        if not self._started:
            text = text.lstrip()
        if text:
            self._started = True
            yield kind(text)
