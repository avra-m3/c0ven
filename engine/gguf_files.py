"""A split GGUF's file names. Only the first shard holds the metadata. The front end has its own copy of the name
rule (frontend/model.py), since it does not import the engine (ADR-0024, item 7)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

SHARD = re.compile(r"^(?P<stem>.*)-\d{5}-of-(?P<count>\d{5})\.gguf$")
"""A shard's file name: `<stem>-NNNNN-of-MMMMM.gguf`."""


def first_shard(model: Path) -> Path:
    """The first shard of a split GGUF, from the path of any of its shards; any other file is itself."""
    m = SHARD.match(model.name)
    return model if m is None else model.with_name(f"{m['stem']}-00001-of-{m['count']}.gguf")


def shard_paths(model: Path) -> tuple[Path, ...]:
    """All shards of a split GGUF, from the path of any of them; any other file is its own list."""
    m = SHARD.match(model.name)
    if m is None:
        return (model,)
    count = int(m["count"])
    return tuple(model.with_name(f"{m['stem']}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1))
