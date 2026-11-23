"""Checkpoint placement for inkling2_loop prompts (ADR-0017, item 6)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engine.models.inkling2_loop.tensors import TensorTable
    from engine.scheduler import CheckpointRule

PERIODIC = 16_384
"""Distance between checkpoints inside one long prompt."""
KEEP_SYSTEM = 2048
"""Minimum length for a system prompt checkpoint to be marked kept."""


@dataclass(frozen=True, slots=True)
class ChatTokens:
    """Special token IDs used by the Inkling-2 chat template."""

    message_system: int
    message_user: int
    message_model: int
    end_message: int
    end_sampling: int

    @classmethod
    def of(cls, table: TensorTable) -> ChatTokens:
        return cls(
            message_system=table.token_id("<|message_system|>"),
            message_user=table.token_id("<|message_user|>"),
            message_model=table.token_id("<|message_model|>"),
            end_message=table.token_id("<|end_message|>"),
            end_sampling=table.token_id("<|content_model_end_sampling|>"),
        )


def checkpoint_rule(tokens: ChatTokens) -> CheckpointRule:
    """Return checkpoint positions `(pos, keep)` for an inkling2_loop prompt."""

    def rule(prompt: tuple[int, ...]) -> list[tuple[int, bool]]:
        n = len(prompt)
        marks: dict[int, bool] = {}

        # Keep a long leading system message right after its <|end_message|>.
        if n > 1 and prompt[0] == tokens.message_system:
            for i in range(1, n):
                if prompt[i] == tokens.end_message:
                    end = i + 1
                    if KEEP_SYSTEM <= end < n:
                        marks[end] = True
                    break

        # Mark each model turn start (skipping position 0).
        for i in range(1, n):
            if prompt[i] == tokens.message_model:
                marks.setdefault(i, False)

        # Periodic marks for long prompts.
        for pos in range(PERIODIC, n, PERIODIC):
            marks.setdefault(pos, False)

        return sorted(marks.items())

    return rule
