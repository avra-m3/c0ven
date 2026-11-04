"""The GGUF tensor table: every tensor's name, type, shard, offset and size (ADR-0004: read with llama.cpp's `gguf`).

The control plane reads the table and passes offsets and types to the core in `open`; the core never parses the
file. A TSV's first line holds the file's `context_length`. The committed docs/model/tensors-IQ3_S.tsv holds the same
rows for the IQ3_S file, so tests run without it; tensors-UD-IQ4_XS.tsv holds them for Unsloth's UD-IQ4_XS file, an
experiment: `from_tsv(TSV_UD_IQ4_XS)`.
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

TABLES = Path(__file__).resolve().parents[3] / "docs" / "model"
TSV = TABLES / "tensors-IQ3_S.tsv"
CONTEXT_LINE = "# context_length "
TSV_UD_IQ4_XS = TABLES / "tensors-UD-IQ4_XS.tsv"
_LAYER = re.compile(r"^blk\.(\d+)\.")


class Category(StrEnum):
    """The parts of MODEL_FILES.md, "Bytes per type and part"."""

    EXPERTS = "experts"
    PLE = "ple"
    DENSE = "dense"
    EMBEDDING = "embedding"
    HEAD = "head"
    NGRAM = "ngram"


class LayerKind(StrEnum):
    ATTENTION = "attn"
    DELTANET = "gdn"


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
        if re.fullmatch(r"blk\.\d+\.ple_.*", name):
            return Category.PLE
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
    context_length: int = 0
    """The context the model was trained on, from the file; 0 in a table kept before it was read."""

    def __post_init__(self) -> None:
        names = [t.name for t in self.tensors]
        if len(set(names)) != len(names):
            raise ValueError("duplicate tensor names")

    def by_name(self) -> dict[str, FileTensor]:
        return {t.name: t for t in self.tensors}

    def _shape_of(self, name: str) -> tuple[int, ...]:
        for t in self.tensors:
            if t.name == name:
                return t.shape
        raise ValueError(f"tensor missing from the model: {name}")

    @property
    def layers(self) -> int:
        """The file's layers: one past the highest numbered block."""
        return 1 + max(t.layer for t in self.tensors)

    @property
    def vocab(self) -> int:
        """The rows of output.weight, as the core reads it (core/models/qwen4exp/shape.h)."""
        return self._shape_of("output.weight")[1]

    @property
    def experts_per_layer(self) -> int:
        return self._shape_of("blk.0.ffn_gate_exps.weight")[2]

    def layer_kinds(self) -> list[LayerKind]:
        names = self.by_name()
        kinds: list[LayerKind] = []
        for n in range(self.layers):
            if f"blk.{n}.attn_k.weight" in names:
                kinds.append(LayerKind.ATTENTION)
            elif f"blk.{n}.ssm_a" in names:
                kinds.append(LayerKind.DELTANET)
            else:
                raise ValueError(f"layer {n} is neither attention nor DeltaNet")
        return kinds

    def bytes_of(self, category: Category) -> int:
        return sum(t.nbytes for t in self.tensors if t.category is category)

    def expert_bytes(self, layer: int) -> int:
        """One routed expert of the layer: (gate + up + down) / the experts in a layer."""
        total = sum(t.nbytes for t in self.tensors if t.layer == layer and t.category is Category.EXPERTS)
        return total // self.experts_per_layer


def from_tsv(path: Path = TSV) -> TensorTable:
    rows: list[FileTensor] = []
    context = 0
    with path.open(newline="") as f:
        first = f.readline()
        if first.startswith(CONTEXT_LINE):
            context = int(first.removeprefix(CONTEXT_LINE))
        else:
            f.seek(0)
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
    return TensorTable(tuple(rows), context_length=context)


def from_gguf(path: Path) -> TensorTable:
    paths = shard_paths(path)
    rows: list[FileTensor] = []
    context = 0
    for shard, p in enumerate(paths, start=1):
        reader = gguf.GGUFReader(p)
        if shard == 1:
            field = reader.get_field("qwen4exp.context_length")
            if field is None:
                raise ValueError(f"{p} has no qwen4exp.context_length")
            context = int(field.contents())
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
    return TensorTable(tuple(rows), paths, context)


def to_tsv(table: TensorTable, path: Path) -> None:
    """The table in the committed file's form, which `from_tsv` reads."""
    with path.open("w", newline="") as f:
        f.write(f"{CONTEXT_LINE}{table.context_length}\n")
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["name", "type", "ggml_type", "shape", "bytes", "shard", "data_offset", "category"])
        for t in table.tensors:
            shape = "x".join(map(str, t.shape))
            w.writerow([t.name, t.type_name, t.ggml_type, shape, t.nbytes, t.shard, t.offset, t.category])


def from_gguf_cached(path: Path, cache: Path | None) -> TensorTable:
    """`from_gguf`, kept in `cache` as a TSV named by each shard's path, size and modification time. A start after
    the first reads the TSV, much faster than the GGUF's header."""
    if cache is None:
        return from_gguf(path)
    paths = shard_paths(path)
    key = "\n".join(f"{p.resolve()} {p.stat().st_size} {p.stat().st_mtime_ns}" for p in paths)
    file = cache / f"tensors-{hashlib.sha256(key.encode()).hexdigest()[:16]}.tsv"
    if file.is_file():
        kept = from_tsv(file)
        if kept.context_length:
            return replace(kept, shard_paths=paths)
    table = from_gguf(path)
    cache.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
    to_tsv(table, tmp)
    tmp.replace(file)
    return table
