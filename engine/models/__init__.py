"""The model architectures. Each `models/<arch>/setup.py` is a `ModelSetup`; the engine process finds it by the GGUF's
`general.architecture` (ADR-0024, item 3; ADR-0060)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    from engine.api.interface import OpenConfig
    from engine.memory_plan import MemoryPlan, PlanOptions
    from engine.scheduler import CheckpointRule


class ModelSetup(Protocol):
    """What the engine process and `tools/plan.py` take from a model's setup module."""

    def open_config(
        self,
        model: Path | None,
        options: PlanOptions,
        deterministic: bool = False,
        draft: Path | None = None,
        table_cache: Path | None = None,
        *,
        cpu_workers: int = ...,
    ) -> tuple[OpenConfig, MemoryPlan]:
        """The config for `open` and the memory plan it holds. model: any shard of the GGUF, or None for the mock and
        the reference, which need no tensors. draft: the drafter's path, or None for no drafting. table_cache: a
        directory keeping the tensor table between starts, or None. cpu_workers: the CPU's expert workers. Raises
        PlanError when the options do not fit."""
        ...

    def draft_dir(self, model: Path) -> Path | None:
        """The drafter the model finds beside `model`, a directory or a file; None: there is none."""
        ...

    def vision_file(self, model: Path) -> Path | None:
        """The image encoder's file the model finds beside `model`; None: there is none."""
        ...

    def checkpoint_rule(self, model: Path | None, table_cache: Path | None = None) -> CheckpointRule:
        """Where to checkpoint a prompt of the model's token ids: (position, kept) pairs. model and table_cache: as
        `open_config`'s."""
        ...
