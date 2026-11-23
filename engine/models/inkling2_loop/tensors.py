"""The inkling2_loop tensor table and metadata, from a GGUF file or the committed Q8_0 census."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

import gguf

from engine.gguf_files import first_shard, shard_paths

TSV = Path(__file__).resolve().parents[3] / "docs" / "model" / "tensors-inkling2-Q8_0.tsv"
"""Committed Q8_0 census: used when the engine runs without a model file (mock or reference)."""

_BLOCK = re.compile(r"^blk\.(\d+)\.(.+)$")

_SPECIAL_TOKENS = (
    "<|message_system|>",
    "<|message_user|>",
    "<|message_model|>",
    "<|message_tool|>",
    "<|end_message|>",
    "<|content_text|>",
    "<|content_thinking|>",
    "<|content_invoke_tool_json|>",
    "<|content_model_end_sampling|>",
    "<|content_image|>",
    "<|unused_200054|>",
    "\u010a",
)


class Category(StrEnum):
    HEAD = "head"
    EMBEDDING = "embedding"
    DENSE = "dense"
    EXPERTS = "experts"


def _category(name: str) -> Category:
    if name in ("output.weight", "output_norm.weight", "output_res_query.weight"):
        return Category.HEAD
    if name == "token_embd.weight":
        return Category.EMBEDDING
    if "_exps." in name:
        return Category.EXPERTS
    return Category.DENSE


@dataclass(frozen=True, slots=True)
class FileTensor:
    """One tensor in a GGUF file, with ggml dimensions (ne0 first)."""

    name: str
    type: str
    ggml_type: int
    shape: tuple[int, ...]
    nbytes: int
    shard: int
    data_offset: int
    category: Category

    @property
    def layer(self) -> int | None:
        m = _BLOCK.match(self.name)
        return int(m.group(1)) if m else None

    @property
    def suffix(self) -> str:
        m = _BLOCK.match(self.name)
        return m.group(2) if m else self.name


def _as_int(value: object, key: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{key} is a boolean, expected an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, list):
        items = cast("list[object]", value)
        if items and all(isinstance(x, int) and not isinstance(x, bool) for x in items):
            ints = cast("list[int]", items)
            first = ints[0]
            if any(x != first for x in ints):
                raise ValueError(f"{key} varies across layers: {value}")
            return first
    raise ValueError(f"{key} is not an integer: {value!r}")


def _as_float(value: object, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} is not a number: {value!r}")
    return float(value)


def _as_bools(value: object, key: str) -> tuple[bool, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{key} is not a boolean array: {value!r}")
    items = cast("list[object]", value)
    if not all(isinstance(x, bool) for x in items):
        raise ValueError(f"{key} is not a boolean array: {value!r}")
    return tuple(cast("list[bool]", items))


@dataclass(frozen=True, slots=True)
class TensorTable:
    """An inkling2_loop model's tensors and GGUF metadata."""

    tensors: tuple[FileTensor, ...]
    shard_paths: tuple[Path, ...] = ()
    fields: dict[str, object] = field(default_factory=dict)

    def by_name(self) -> dict[str, FileTensor]:
        return {t.name: t for t in self.tensors}

    def field(self, key: str) -> object:
        if key not in self.fields:
            raise KeyError(f"GGUF metadata key missing: {key}")
        return self.fields[key]

    @property
    def architecture(self) -> str:
        return str(self.field("general.architecture"))

    def arch_int(self, suffix: str) -> int:
        key = f"{self.architecture}.{suffix}"
        return _as_int(self.field(key), key)

    def arch_float(self, suffix: str) -> float:
        key = f"{self.architecture}.{suffix}"
        return _as_float(self.field(key), key)

    @property
    def layers(self) -> int:
        """Physical layer count (`block_count`)."""
        return self.arch_int("block_count")

    @property
    def loop_start(self) -> int:
        return self.arch_int("loop.start")

    @property
    def loop_end(self) -> int:
        return self.arch_int("loop.end")

    @property
    def loop_count(self) -> int:
        return self.arch_int("loop.count")

    @property
    def logical_layers(self) -> int:
        """Logical KV and convolution slots across all passes."""
        return self.layers + (self.loop_count - 1) * (self.loop_end - self.loop_start)

    @property
    def swa_pattern(self) -> tuple[bool, ...]:
        key = f"{self.architecture}.attention.sliding_window_pattern"
        pat = _as_bools(self.field(key), key)
        if len(pat) != self.layers:
            raise ValueError(f"{key} has {len(pat)} entries for {self.layers} physical layers")
        return pat

    @property
    def logical_swa_pattern(self) -> tuple[bool, ...]:
        phys = self.swa_pattern
        span = self.loop_end - self.loop_start
        return tuple(
            phys[il] if il < self.layers else phys[self.loop_start + (il - self.layers) % span]
            for il in range(self.logical_layers)
        )

    @property
    def global_layers(self) -> int:
        return sum(1 for swa in self.logical_swa_pattern if not swa)

    @property
    def swa_layers(self) -> int:
        return sum(1 for swa in self.logical_swa_pattern if swa)

    @property
    def context_length(self) -> int:
        return self.arch_int("context_length")

    @property
    def embedding_length(self) -> int:
        return self.arch_int("embedding_length")

    @property
    def head_count(self) -> int:
        return self.arch_int("attention.head_count")

    @property
    def head_count_kv(self) -> int:
        return self.arch_int("attention.head_count_kv")

    @property
    def head_dim(self) -> int:
        return self.arch_int("attention.key_length")

    @property
    def sliding_window(self) -> int:
        return self.arch_int("attention.sliding_window")

    @property
    def shortconv_kernel(self) -> int:
        return self.arch_int("shortconv.l_cache")

    @property
    def experts_per_layer(self) -> int:
        return self.arch_int("expert_count")

    @property
    def experts_used(self) -> int:
        return self.arch_int("expert_used_count")

    @property
    def expert_ff(self) -> int:
        return self.arch_int("expert_feed_forward_length")

    @property
    def vocab_size(self) -> int:
        """Rows of `output.weight`."""
        return self.by_name()["output.weight"].shape[1]

    @property
    def vocab_unpadded(self) -> int:
        key = f"{self.architecture}.vocab_size_unpadded"
        if key in self.fields:
            return _as_int(self.fields[key], key)
        return self.vocab_size

    def token_id(self, token: str) -> int:
        ids = cast("dict[str, int]", self.field("c0ven.token_ids"))
        if token not in ids:
            raise KeyError(f"token {token!r} missing from c0ven.token_ids")
        return int(ids[token])


def _read_metadata(reader: gguf.GGUFReader) -> dict[str, object]:
    skip = {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.scores",
        "tokenizer.chat_template",
    }
    meta: dict[str, object] = {}
    for key, fld in reader.fields.items():
        if key.startswith("GGUF.") or key in skip:
            continue
        val = fld.contents()
        if isinstance(val, list):
            items = cast("list[object]", val)
            meta[key] = [bool(x) if isinstance(x, bool) else x for x in items]
        else:
            meta[key] = val
    tok_field = reader.get_field("tokenizer.ggml.tokens")
    if tok_field is not None:
        vocab = cast("list[str]", tok_field.contents())
        index = {tok: i for i, tok in enumerate(vocab) if tok in _SPECIAL_TOKENS}
        meta["c0ven.token_ids"] = index
    return meta


def from_tsv(path: Path = TSV) -> TensorTable:
    """Read a committed census TSV and its sibling metadata JSON."""
    rows: list[FileTensor] = []
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["name", "type", "ggml_type", "shape", "bytes", "shard", "data_offset", "category"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            name, tname, gtype, shape_s, nbytes, shard, offset, cat = line.split("\t")
            rows.append(
                FileTensor(
                    name=name,
                    type=tname,
                    ggml_type=int(gtype),
                    shape=tuple(int(d) for d in shape_s.split("x")),
                    nbytes=int(nbytes),
                    shard=int(shard),
                    data_offset=int(offset),
                    category=Category(cat),
                )
            )
    meta_path = path.with_suffix(".json")
    fields = cast("dict[str, object]", json.loads(meta_path.read_text(encoding="utf-8"))) if meta_path.is_file() else {}
    return TensorTable(tuple(rows), (), fields)


def from_gguf(path: Path) -> TensorTable:
    """Read the tensor table and metadata from a GGUF file (any shard)."""
    shards = shard_paths(first_shard(path))
    rows: list[FileTensor] = []
    fields: dict[str, object] = {}
    for idx, shard in enumerate(shards, start=1):
        reader = gguf.GGUFReader(shard)
        if idx == 1:
            fields = _read_metadata(reader)
        for t in reader.tensors:
            rows.append(
                FileTensor(
                    name=t.name,
                    type=t.tensor_type.name,
                    ggml_type=int(t.tensor_type),
                    shape=tuple(int(d) for d in t.shape),
                    nbytes=int(t.n_bytes),
                    shard=idx,
                    data_offset=int(t.data_offset),
                    category=_category(t.name),
                )
            )
    return TensorTable(tuple(rows), shards, fields)


def _cache_key(shards: tuple[Path, ...]) -> str:
    parts: list[str] = []
    for p in shards:
        st = p.stat()
        parts.append(f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def from_gguf_cached(path: Path, cache: Path | None) -> TensorTable:
    """Read a GGUF's tensor table, caching it in `cache` keyed by shard paths, sizes and mtimes."""
    shards = shard_paths(first_shard(path))
    if cache is None:
        return from_gguf(path)
    key = _cache_key(shards)
    file = cache / f"inkling2-table-{key}.json"
    if file.is_file():
        try:
            raw = cast("dict[str, Any]", json.loads(file.read_text(encoding="utf-8")))
            rows = tuple(
                FileTensor(
                    name=r["name"],
                    type=r["type"],
                    ggml_type=int(r["ggml_type"]),
                    shape=tuple(int(d) for d in r["shape"]),
                    nbytes=int(r["nbytes"]),
                    shard=int(r["shard"]),
                    data_offset=int(r["data_offset"]),
                    category=Category(r["category"]),
                )
                for r in raw["tensors"]
            )
            return TensorTable(rows, shards, cast("dict[str, object]", raw["fields"]))
        except OSError, ValueError, KeyError, TypeError:
            pass
    table = from_gguf(path)
    try:
        cache.mkdir(parents=True, exist_ok=True)
        payload = {
            "tensors": [
                {
                    "name": t.name,
                    "type": t.type,
                    "ggml_type": t.ggml_type,
                    "shape": list(t.shape),
                    "nbytes": t.nbytes,
                    "shard": t.shard,
                    "data_offset": t.data_offset,
                    "category": t.category.value,
                }
                for t in table.tensors
            ],
            "fields": table.fields,
        }
        tmp = file.with_name(f"{file.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(file)
    except OSError:
        pass
    return table
