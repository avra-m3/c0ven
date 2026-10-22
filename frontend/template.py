"""The chat template (ADR-0004: `jinja2`, with the model's own template).

The template is the GGUF's `tokenizer.chat_template` or the model's `chat_template.jinja`, rendered as Hugging Face's
`apply_chat_template` renders it: a sandbox, `trim_blocks` and `lstrip_blocks`, a `tojson` that keeps non-ASCII text
and does not escape HTML, and `raise_exception`. What the template does is in spec/MODEL_IO.md, section 4.2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, NoReturn

from jinja2 import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

type JSON = Any


class TemplateFailed(ValueError):
    """The template refused the conversation (its own `raise_exception`, or a render error)."""


def _raise(message: str) -> NoReturn:
    raise TemplateFailed(message)


def _tojson(value: JSON, indent: int | None = None, ensure_ascii: bool = False, sort_keys: bool = False) -> str:
    return json.dumps(value, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys)


@dataclass(frozen=True, slots=True)
class Conversation:
    """What the template is given. `messages` and `tools` are in the OpenAI shape the template expects."""

    messages: list[dict[str, JSON]]
    tools: list[dict[str, JSON]] | None = None
    enable_thinking: bool | None = None
    reasoning_effort: str | None = None
    preserve_thinking: bool | None = None
    add_generation_prompt: bool = True


class ChatTemplate:
    def __init__(self, source: str) -> None:
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise  # pyright: ignore[reportArgumentType]
        self.source = source
        self._template = env.from_string(source)

    def render(self, c: Conversation) -> str:
        variables: dict[str, JSON] = {"messages": c.messages, "add_generation_prompt": c.add_generation_prompt}
        if c.tools:
            variables["tools"] = c.tools
        # Undefined and false differ to the template ("enable_thinking is undefined or enable_thinking is true"),
        # so a setting the client did not give is left out.
        for key in ("enable_thinking", "reasoning_effort", "preserve_thinking"):
            value = getattr(c, key)
            if value is not None:
                variables[key] = value
        try:
            return self._template.render(**variables)
        except TemplateFailed:
            raise
        except TemplateError as exc:
            raise TemplateFailed(str(exc)) from None
