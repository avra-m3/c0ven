"""The chat template (ADR-0004: `jinja2`, with the model's own template).

The template is the GGUF's `tokenizer.chat_template` or the model's `chat_template.jinja`, rendered as Hugging Face's
`apply_chat_template` renders it: a sandbox, `trim_blocks` and `lstrip_blocks`, a `tojson` that keeps non-ASCII text
and does not escape HTML, and `raise_exception`. What the template does is in spec/MODEL_IO.md, section 4.2. The
reasoning efforts it accepts are found by rendering one message with each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, NoReturn

from jinja2 import TemplateError, meta
from jinja2.sandbox import ImmutableSandboxedEnvironment

type JSON = Any

LEVELS = ("minimal", "low", "medium", "high", "xhigh", "max")
"""The reasoning-effort levels a template is tried with, lowest first."""
_PROBE: list[dict[str, JSON]] = [{"role": "user", "content": "Hi"}]


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
    reasoning_effort: str | int | None = None
    preserve_thinking: bool | None = None
    add_generation_prompt: bool = True


@dataclass(frozen=True, slots=True)
class Efforts:
    """The reasoning efforts a template accepts: `levels`, some of `LEVELS` in their order, and `integers`, whether it
    accepts an integer."""

    levels: tuple[str, ...]
    integers: bool

    def nearest(self, level: str) -> str | None:
        """The accepted level nearest to `level`, one of `LEVELS`; of two as near, the higher. None when the template
        accepts no level."""
        at = LEVELS.index(level)
        return max(self.levels, key=lambda a: (-abs(LEVELS.index(a) - at), LEVELS.index(a)), default=None)

    def given(self, value: object) -> str | int | None:
        """A client's own `chat_template_kwargs.reasoning_effort`: a string as it is, an integer only when the template
        accepts integers, anything else left out."""
        if isinstance(value, str):
            return value
        if isinstance(value, int) and not isinstance(value, bool) and self.integers:
            return value
        return None


class ChatTemplate:
    def __init__(self, source: str) -> None:
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise  # pyright: ignore[reportArgumentType]
        self.source = source
        parsed = env.parse(source)
        self._variables = meta.find_undeclared_variables(parsed)
        self._template = env.from_string(parsed)
        self.efforts = self._efforts()
        """The reasoning efforts the template accepts."""

    def _accepts(self, effort: str | int | None) -> bool:
        try:
            self.render(Conversation(_PROBE, enable_thinking=True, reasoning_effort=effort))
        except Exception:  # Any failure means not accepted, so the probe never stops a load.
            return False
        return True

    def _efforts(self) -> Efforts:
        if not self._accepts(None):
            # The template refuses the probe's message itself, so the probe tells nothing: each request's effort goes
            # to the template as asked.
            return Efforts(LEVELS, integers=True)
        return Efforts(tuple(level for level in LEVELS if self._accepts(level)), self._accepts(1))

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
        # A template with `drop_thinking` (default true) drops earlier turns' reasoning unless told not to.
        if c.preserve_thinking is not None and "drop_thinking" in self._variables:
            variables["drop_thinking"] = not c.preserve_thinking
        try:
            return self._template.render(**variables)
        except TemplateFailed:
            raise
        except TemplateError as exc:
            raise TemplateFailed(str(exc)) from None
