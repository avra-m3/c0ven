"""The GGUF tensor table: every tensor's name, type, shard, offset and size (ADR-0004: read with llama.cpp's `gguf`).

The control plane reads the table and passes offsets and types to the core in `open`; the core never parses the file.
tensors.tsv holds the same rows for Unsloth's UD-IQ2_XXS file, so tests run without it.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

import gguf

from engine.gguf_files import shard_paths
from engine.models.deepseek41 import numbers

TSV = Path(__file__).with_name("tensors.tsv")
_LAYER = re.compile(r"^blk\.(\d+)\.")


class Category(StrEnum):
    EXPERTS = "experts"
    DENSE = "dense"
    EMBEDDING = "embedding"
    HEAD = "head"
    NGRAM = "ngram"
    """The Engram table."""


@dataclass(frozen=True, slots=True)
class FileTensor:
    name: str
    type_name: str
    ggml_type: int
    shape: tuple[int, ...]
    """In ggml's order: ne0 (the row length) first."""
    nbytes: int
    shard: int
    """Counted from 1, as the file names do."""
    offset: int
    """From the start of its shard: the data section's start plus the tensor's own offset."""

    @property
    def layer(self) -> int:
        m = _LAYER.match(self.name)
        return int(m.group(1)) if m else -1

    @property
    def category(self) -> Category:
        name = self.name
        if re.fullmatch(r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight", name):
            return Category.EXPERTS
        if name.startswith("blk."):
            return Category.DENSE
        if name == "token_embd.weight":
            return Category.EMBEDDING
        if name == "per_layer_token_embd.weight":
            return Category.NGRAM
        if name.startswith("output"):
            return Category.HEAD
        raise ValueError(f"tensor outside every part: {name}")


@dataclass(frozen=True, slots=True)
class TensorTable:
    tensors: tuple[FileTensor, ...]
    shard_paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        names = [t.name for t in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tensor names")

    def by_name(self) -> dict[str, FileTensor]:
        return {t.name: t for t in self.tensors}

    def bytes_of(self, category: Category) -> int:
        return sum(t.nbytes for t in self.tensors if t.category is category)

    def expert_bytes(self, layer: int) -> int:
        """One routed expert of the layer: (gate + up + down) / 384."""
        total = sum(t.nbytes for t in self.tensors if t.layer == layer and t.category is Category.EXPERTS)
        return total // numbers.EXPERTS_PER_LAYER


def from_tsv(path: Path = TSV) -> TensorTable:
    """The table in the file at `path`, written by `to_tsv`, with no shard paths. Its category column is not read: a
    tensor's name gives its category. Raises OSError, and KeyError or ValueError for a file of another form."""
    rows: list[FileTensor] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rows.append(
                FileTensor(
                    row["name"],
                    row["type"],
                    int(row["ggml_type"]),
                    tuple(int(d) for d in row["shape"].split("x")),
                    int(row["bytes"]),
                    int(row["shard"]),
                    int(row["data_offset"]),
                )
            )
    return TensorTable(tuple(rows))


def from_gguf(path: Path) -> TensorTable:
    """The table of the GGUF at `path`, any of its shards, read from every shard's header. Raises OSError, and
    ValueError for a file that is not a GGUF."""
    paths = shard_paths(path)
    rows: list[FileTensor] = []
    for shard, p in enumerate(paths, start=1):
        reader = gguf.GGUFReader(p)
        for t in reader.tensors:
            kind = gguf.GGMLQuantizationType(int(t.tensor_type))
            rows.append(
                FileTensor(
                    t.name,
                    kind.name,
                    int(kind),
                    tuple(int(d) for d in t.shape),
                    int(t.n_bytes),
                    shard,
                    int(t.data_offset),
                )
            )
    return TensorTable(tuple(rows), paths)


def to_tsv(table: TensorTable, path: Path) -> None:
    """Writes `table` to `path` in the committed file's form, which `from_tsv` reads. Its shard paths are not written.
    Raises ValueError for a tensor outside every category."""
    with path.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["name", "type", "ggml_type", "shape", "bytes", "shard", "data_offset", "category"])
        for t in table.tensors:
            shape = "x".join(map(str, t.shape))
            w.writerow([t.name, t.type_name, t.ggml_type, shape, t.nbytes, t.shard, t.offset, t.category])


def from_gguf_cached(path: Path, cache: Path | None) -> TensorTable:
    """`from_gguf`, kept in the directory `cache` as a TSV named by each shard's path, size and modification time. A
    start after the first reads the TSV, much faster than the GGUF's header. cache: None reads the GGUF each time.
    Raises as `from_gguf` does."""
    if cache is None:
        return from_gguf(path)
    paths = shard_paths(path)
    key = "\n".join(f"{p.resolve()} {p.stat().st_size} {p.stat().st_mtime_ns}" for p in paths)
    file = cache / f"tensors-{hashlib.sha256(key.encode()).hexdigest()[:16]}.tsv"
    if file.is_file():
        return replace(from_tsv(file), shard_paths=paths)
    table = from_gguf(path)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    to_tsv(table, tmp)
    tmp.replace(file)
    return table
