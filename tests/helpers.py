"""What the tests share: a tiny tokenizer and chat template for the front end, engine configs for the mock, and a
fake engine. The model's own tokenizer and template are tested in tests/end_to_end/, where the model's files are."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field

from tokenizers import Tokenizer, pre_tokenizers

import protocol
from engine.api.interface import EngineFacts, LoadCheck, OpenConfig
from frontend.engine_client import EngineUnavailable
from frontend.images import PromptImage
from frontend.model import Model
from frontend.template import ChatTemplate
from frontend.tokenizer import CONTROL, USER_DEFINED, encode, from_vocabulary

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
ADDED = ["<think>", "</think>", "<tool_call>", "</tool_call>"]
VISION = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"]

TINY_TEMPLATE = (
    "{%- if tools %}{{- '<|im_start|>system\\nTOOLS' }}{%- for t in tools %}{{- '\\n' + (t | tojson) }}{%- endfor %}"
    "{{- '<|im_end|>\\n' }}{%- endif %}"
    "{%- for m in messages %}"
    "{{- '<|im_start|>' + m.role + '\\n' }}"
    "{%- if m.reasoning_content %}{{- '<think>\\n' + m.reasoning_content + '\\n</think>\\n\\n' }}{%- endif %}"
    "{%- if m.content is string %}{{- m.content }}{%- else %}{%- for item in m.content %}"
    "{%- if item.type == 'image' %}{{- '<|vision_start|><|image_pad|><|vision_end|>' }}"
    "{%- elif item.type == 'video' %}{{- '<|vision_start|><|video_pad|><|vision_end|>' }}{%- else %}{{- item.text }}"
    "{%- endif %}{%- endfor %}{%- endif %}"
    "{%- for c in m.tool_calls or [] %}{{- '<tool_call>\\n<function=' + c.function.name + '>\\n' }}"
    "{%- for k, v in c.function.arguments|items %}{{- '<parameter=' + k + '>\\n' + (v|string) + '\\n</parameter>\\n' }}"
    "{%- endfor %}{{- '</function>\\n</tool_call>' }}{%- endfor %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>\\n' }}{%- endif %}{%- endif %}"
)
"""A ChatML template with the model's thinking and tool-call formats, small enough to read in a test."""


def tiny_tokenizer() -> Tokenizer:
    """One token per byte (ids 0-255), then the special and added tokens: every UTF-8 character of more than one
    byte spans several tokens, which is what the detokenizer must handle."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    tokens = [*alphabet, *SPECIALS, *ADDED, *VISION]
    types = [1] * len(alphabet) + [CONTROL] * len(SPECIALS) + [USER_DEFINED] * len(ADDED) + [CONTROL] * len(VISION)
    return from_vocabulary(tokens, [], types)


def tiny_model(tokenizer: Tokenizer) -> Model:
    stop = frozenset(i for t in ("<|im_end|>", "<|endoftext|>") if (i := tokenizer.token_to_id(t)) is not None)
    return Model("tiny", tokenizer, ChatTemplate(TINY_TEMPLATE), stop, protocol.Settings(), "tiny.gguf")


def open_config(slots: int = 2, context: int = 4096, pages: int | None = None, checkpoints: int = 4) -> OpenConfig:
    return OpenConfig(
        shard_paths=(),
        tensors=(),
        slots=slots,
        max_context=context,
        pages=pages if pages is not None else slots * (context // 4) + context // 4,
        checkpoints=checkpoints,
        kv_resident_cells=0,
        experts_cached=0,
        arena_bytes=0,
        working_bytes=0,
        cpu_workers=0,
        ngram_cache_rows=0,
        timing_records=0,
        eos_ids=(2,),
        vocab_size=1000,
    )


def facts_of(config: OpenConfig) -> EngineFacts:
    return EngineFacts(0, config.slots, config.max_context, config.pages, config.checkpoints, 0, (), (), 0, -1, "t")


@dataclass
class FakeEngine:
    """The front end's view of an engine, answering every request with `output`, encoded with the tokenizer."""

    tokenizer: Tokenizer
    output: str = "Hello."
    max_context: int = 4096
    finish: protocol.Finish = protocol.Finish.STOP
    eos: str | None = "<|im_end|>"
    reused: int = 0
    prompts: list[list[int]] = field(default_factory=lambda: [])
    max_news: list[int] = field(default_factory=lambda: [])
    """The max_new each generation was given."""
    stopped_early: int = 0
    hang: bool = False
    """After its tokens, wait for a stop instead of finishing."""
    image_tokens_max: int = 0
    images: list[list[PromptImage]] = field(default_factory=lambda: [])
    status: str = "ready"
    wake_error: str | None = None
    sleeps: list[int] = field(default_factory=lambda: [])
    """The drain each sleep was given, in ms."""
    wakes: int = 0
    counts: dict[str, str] = field(default_factory=lambda: {"plans": "4", "steps": "3", "step_ns": "30000000"})
    """Its STATS line."""

    @property
    def info(self) -> dict[str, str]:
        return {"backend": "fake"}

    @property
    def slots(self) -> int:
        return 1

    @property
    def busy(self) -> int:
        return 0

    @property
    def asleep(self) -> bool:
        return self.status in ("loading", "sleeping", "asleep")

    async def start(self) -> None:
        pass

    async def wake(self) -> dict[str, str]:
        if self.wake_error is not None:
            raise EngineUnavailable(self.wake_error)
        self.wakes += 1
        self.status = "ready"
        return self.info

    async def sleep(self, drain_ms: int) -> float:
        self.sleeps.append(drain_ms)
        self.status = "asleep"
        return 1.5

    async def close(self) -> None:
        pass

    async def stats(self) -> dict[str, str]:
        return dict(self.counts)

    async def steps(self) -> list[dict[str, str]]:
        return [{"kind": "decode", "plan": "7", "rows": "1", "ns": "15500000"}]

    async def generate(
        self,
        prompt: Sequence[int],
        max_new: int,
        settings: protocol.Settings,
        images: Sequence[PromptImage] = (),
    ) -> AsyncGenerator[protocol.Event]:
        self.prompts.append(list(prompt))
        self.max_news.append(max_new)
        self.images.append(list(images))
        ids = encode(self.tokenizer, self.output)
        if self.eos is not None and self.finish is protocol.Finish.STOP:
            eos = self.tokenizer.token_to_id(self.eos)
            assert eos is not None
            ids.append(eos)
        ids = ids[:max_new]
        finish = protocol.Finish.LENGTH if len(ids) == max_new and self.finish is protocol.Finish.STOP else self.finish
        done = False
        try:
            if self.reused:
                yield protocol.Resume(1, self.reused)
            for i in ids:
                yield protocol.Tokens(1, (i,))
            if self.hang:
                await asyncio.Event().wait()
            done = True
            yield protocol.Done(1, len(ids), len(prompt), 1.0, 2.0, finish, self.reused, 0.5, len(ids), 2, 1)
        finally:
            if not done:
                self.stopped_early += 1


def load_check(
    gate_up: dict[int, dict[int, float]] | None = None,
    down: dict[int, dict[int, float]] | None = None,
    dense: dict[int, float] | None = None,
    exchange: dict[int, float] | None = None,
    handoff: float | None = None,
) -> LoadCheck:
    """A load check of the IQ3_S file's kernels with these timings and nothing else."""
    return LoadCheck("IQ3_S", "IQ4_NL", "Q6_K", gate_up or {}, down or {}, dense or {}, exchange or {}, handoff, 1.0)
