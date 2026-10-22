"""The reference: llama.cpp's CPU backend at the pinned commit (ADR-0012), as recorded (REFERENCE_OUTPUTS.md).

It replays `reference/llamacpp-3cf0325-IQ3_S-greedy.jsonl`: the greedy tokens for each prompt of
`reference/prompts.jsonl`. Behind the engine's interface it runs the same plans as the mock and checks them the same
way, so the scheduler and the front end can run end to end against the model's real token ids. A prompt it has no
record of fails with an ERROR event, and so does asking for more tokens than were recorded.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from engine.api.mock import MockEngine

REFERENCE_DIR = Path(__file__).resolve().parents[2] / "reference"
PROMPTS = REFERENCE_DIR / "prompts.jsonl"
GREEDY = REFERENCE_DIR / "llamacpp-3cf0325-IQ3_S-greedy.jsonl"


def load_reference(prompts: Path = PROMPTS, greedy: Path = GREEDY) -> dict[tuple[int, ...], tuple[int, ...]]:
    """Each reference prompt's token ids, mapped to the greedy tokens llama.cpp produced for it."""
    by_id: dict[str, tuple[int, ...]] = {}
    with prompts.open() as f:
        for line in f:
            row = json.loads(line)
            by_id[row["id"]] = tuple(row["tokens"])
    out: dict[tuple[int, ...], tuple[int, ...]] = {}
    with greedy.open() as f:
        for line in f:
            row = json.loads(line)
            if "meta" in row:
                continue
            out[by_id[row["id"]]] = tuple(row["tokens"])
    return out


class ReferenceEngine(MockEngine):
    def __init__(self, records: dict[tuple[int, ...], tuple[int, ...]] | None = None) -> None:
        self._records = load_reference() if records is None else records
        self._prompt_lengths = sorted({len(p) for p in self._records})
        super().__init__(self._next)

    def _next(self, tokens: Sequence[int]) -> int:
        for n in self._prompt_lengths:
            if n > len(tokens):
                break
            generated = self._records.get(tuple(tokens[:n]))
            if generated is None:
                continue
            done = len(tokens) - n
            if tuple(tokens[n:]) != generated[:done]:
                raise KeyError("the sequence left the recorded tokens")
            if done >= len(generated):
                raise KeyError(f"only {len(generated)} tokens were recorded")
            return generated[done]
        raise KeyError("no recorded prompt is a prefix of this sequence")
