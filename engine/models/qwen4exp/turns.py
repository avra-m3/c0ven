"""Where qwen4exp's prompts are checkpointed (ADR-0017, item 6), found from the rendered tokens.

- At the start of each assistant turn: before `<|im_start|>assistant`. The last one is the generation prompt, so
  the next request of the same conversation resumes from it.
- Every 16K prompt tokens.
- At the end of a system message of 2,048 tokens or more, after its `<|im_end|>\\n`. That one is kept.

The chat template puts the effort sentence and the tool list first (spec/MODEL_IO.md, section 4.3), so these
positions are per pair of effort and tools, as ADR-0017 says.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from engine.scheduler import every_16k

if TYPE_CHECKING:
    from engine.models.qwen4exp.tensors import TensorTable
    from engine.scheduler import CheckpointRule

LONG_SYSTEM = 2048


@dataclass(frozen=True, slots=True)
class ChatTokens:
    """The ids the rule looks for, the model file's."""

    im_start: int
    im_end: int
    system: int
    assistant: int
    newline: int

    @classmethod
    def of(cls, table: TensorTable) -> ChatTokens:
        return cls(*(table.token_id(t) for t in ("<|im_start|>", "<|im_end|>", "system", "assistant", "\u010a")))


def checkpoint_rule(chat: ChatTokens) -> CheckpointRule:
    """The checkpoint positions of a prompt of the file whose tokens `chat` holds: (position, kept) pairs."""

    def positions(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
        marks: dict[int, bool] = dict(every_16k(prompt))
        for i in range(len(prompt) - 1):
            if prompt[i] == chat.im_start and prompt[i + 1] == chat.assistant:
                marks[i] = marks.get(i, False)
        if prompt[:2] == (chat.im_start, chat.system):
            try:
                end = prompt.index(chat.im_end) + 1
            except ValueError:
                end = -1
            if end >= 0 and end < len(prompt) and prompt[end] == chat.newline:
                end += 1
            if end >= LONG_SYSTEM:
                marks[end] = True
        return sorted((p, k) for p, k in marks.items() if 0 < p < len(prompt))

    return positions
