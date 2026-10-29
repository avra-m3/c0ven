"""The committed tensor tables, against MODEL_FILES.md and MEMORY_BUDGET.md."""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.api.interface import SplitKind
from engine.models.qwen4exp import tensors
from engine.models.qwen4exp.split import split_of
from engine.models.qwen4exp.tensors import (
    TSV_UD_IQ4_XS,
    Category,
    LayerKind,
    from_gguf_cached,
    from_tsv,
    shard_paths,
)

TABLE = from_tsv()


def test_counts_and_sums():
    assert len(TABLE.tensors) == 1224
    assert TABLE.bytes_of(Category.EXPERTS) == 50_292_326_400
    assert TABLE.bytes_of(Category.DENSE) == 3_576_097_280
    assert TABLE.bytes_of(Category.NGRAM) == 28_800_138_240
    assert sum(t.nbytes for t in TABLE.tensors) == 83_606_638_080


def test_layer_kinds():
    kinds = TABLE.layer_kinds()
    assert [i for i, k in enumerate(kinds) if k is LayerKind.ATTENTION] == list(range(3, 48, 4))
    assert kinds.count(LayerKind.DELTANET) == 36


def test_expert_sizes():
    assert TABLE.expert_bytes(3) == 1_971_200
    assert TABLE.expert_bytes(1) == 1_510_400
    assert TABLE.expert_bytes(47) == 2_662_400


def test_the_ud_iq4_xs_table():
    """Unsloth's UD-IQ4_XS file: the IQ3_S file's tensors and shapes in other types, Q8_0 down experts in five
    layers."""
    ud = from_tsv(TSV_UD_IQ4_XS)
    assert {(t.name, t.shape) for t in ud.tensors} == {(t.name, t.shape) for t in TABLE.tensors}
    assert ud.bytes_of(Category.EXPERTS) == 59_519_795_200
    assert ud.bytes_of(Category.DENSE) == 4_112_616_960
    assert ud.bytes_of(Category.NGRAM) == 28_800_138_240
    assert sum(t.nbytes for t in ud.tensors) == 93_671_559_680
    downs = {t.layer for t in ud.tensors if t.name.endswith(".ffn_down_exps.weight") and t.type_name == "Q8_0"}
    assert downs == {2, 4, 30, 46, 47}
    assert ud.expert_bytes(1) == 2_329_600
    assert ud.expert_bytes(2) == 3_481_600


def test_every_tensor_has_a_split_rule():
    kinds = {t.name: split_of(t.name) for t in TABLE.tensors}
    assert kinds["blk.3.ffn_gate_inp.weight"] is SplitKind.WHOLE
    assert kinds["blk.3.attn_output.weight"] is SplitKind.COLUMNS
    assert kinds["blk.1.ffn_down_exps.weight"] is SplitKind.EXPERT_COLUMNS
    assert kinds["per_layer_token_embd.weight"] is SplitKind.FILE
    assert kinds["output.weight"] is SplitKind.ROWS


def test_shards():
    first = Path("/m/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")
    assert [p.name for p in shard_paths(first)] == [
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf",
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf",
    ]
    assert shard_paths(Path("/m/one.gguf")) == (Path("/m/one.gguf"),)


def test_the_table_is_read_from_the_gguf_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The committed table stands in for the GGUF reader; the shards are empty files, as only their size and time
    name the cache."""
    first = tmp_path / "m-00001-of-00002.gguf"
    for p in shard_paths(first):
        p.write_bytes(b"")
    reads: list[Path] = []

    def from_gguf(path: Path) -> tensors.TensorTable:
        reads.append(path)
        return tensors.TensorTable(from_tsv().tensors, shard_paths(path))

    monkeypatch.setattr(tensors, "from_gguf", from_gguf)
    cache = tmp_path / "cache"
    a = from_gguf_cached(first, cache)
    b = from_gguf_cached(first, cache)
    assert reads == [first]
    assert b == a
    assert b.shard_paths == shard_paths(first)
    first.write_bytes(b"changed")
    from_gguf_cached(first, cache)
    assert reads == [first, first]
