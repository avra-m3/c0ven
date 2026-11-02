"""Where deepseek41's prompts are checkpointed (ADR-0017, item 6), found from the rendered tokens: before each
assistant token (numbers.ASSISTANT), every 16K tokens, and at the end of a long system message, which is kept.
"""

from __future__ import annotations

from engine.models.deepseek41 import numbers
from engine.scheduler import every_16k

LONG_SYSTEM = 2048


def checkpoint_positions(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
    """Where to checkpoint `prompt`, its rendered token ids: (position, kept) in rising order, each position between 1
    and the prompt's last. kept: true for a long system message's end, which outlives the other checkpoints."""
    marks: dict[int, bool] = dict(every_16k(prompt))
    # The last is the generation prompt, where the next request resumes. The template's dropped thinking follows the
    # token, so the text before it stays the same.
    for i, token in enumerate(prompt):
        if token == numbers.ASSISTANT:
            marks[i] = marks.get(i, False)
    # The template ends a system message with no token of its own: the first user token is its end.
    if numbers.USER in prompt and (end := prompt.index(numbers.USER)) >= LONG_SYSTEM:
        marks[end] = True
    return sorted((p, k) for p, k in marks.items() if 0 < p < len(prompt))
