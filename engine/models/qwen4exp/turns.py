"""Where qwen4exp's prompts are checkpointed (ADR-0017, item 6), found from the rendered tokens.

- At the start of each assistant turn: before `<|im_start|>assistant`. The last one is the generation prompt, so
  the next request of the same conversation resumes from it.
- Every 16K prompt tokens.
- At the end of a system message of 2,048 tokens or more, after its `<|im_end|>\\n`. That one is kept.

The chat template puts the effort sentence and the tool list first (spec/MODEL_IO.md, section 4.3), so these
positions are per pair of effort and tools, as ADR-0017 says.
"""

from __future__ import annotations

from engine.models.qwen4exp import numbers
from engine.scheduler import every_16k

LONG_SYSTEM = 2048


def checkpoint_positions(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
    marks: dict[int, bool] = dict(every_16k(prompt))
    for i in range(len(prompt) - 1):
        if prompt[i] == numbers.IM_START and prompt[i + 1] == numbers.ASSISTANT:
            marks[i] = marks.get(i, False)
    if prompt[:2] == (numbers.IM_START, numbers.SYSTEM):
        try:
            end = prompt.index(numbers.IM_END) + 1
        except ValueError:
            end = -1
        if end >= 0 and end < len(prompt) and prompt[end] == numbers.NEWLINE:
            end += 1
        if end >= LONG_SYSTEM:
            marks[end] = True
    return sorted((p, k) for p, k in marks.items() if 0 < p < len(prompt))
