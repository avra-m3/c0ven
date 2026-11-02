"""The committed tensor table and its split rules, against the file's sizes."""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import gguf
import numpy as np
import pytest

from engine.api.interface import SplitKind
from engine.gguf_files import shard_paths
from engine.models.deepseek41 import numbers, tensors
from engine.models.deepseek41.split import bytes_on_card, split_of
from engine.models.deepseek41.tensors import TSV, Category, from_gguf, from_gguf_cached, from_tsv, to_tsv

TABLE = from_tsv()
NAMES = TABLE.by_name()


def test_counts_and_sums():
    assert len(TABLE.tensors) == 1_043
    assert TABLE.bytes_of(Category.EXPERTS) == 114_555_617_280
    assert TABLE.bytes_of(Category.DENSE) == 5_249_577_408
    assert TABLE.bytes_of(Category.NGRAM) == 110_595_290_400
    assert NAMES["output.weight"].nbytes == 455_065_600
    assert NAMES["token_embd.weight"].nbytes == 455_065_600
    assert sum(t.nbytes for t in TABLE.tensors) == 231_310_636_768


def test_each_shard_starts_at_the_files_offset_and_each_tensor_ends_where_the_next_starts():
    first = {2: 22_112, 3: 21_632, 4: 20_896, 5: 1_024, 6: 416}
    for shard, offset in first.items():
        rows = sorted((t for t in TABLE.tensors if t.shard == shard), key=lambda t: t.offset)
        assert rows[0].offset == offset, shard
        for a, b in pairwise(rows):
            assert -(-(a.offset + a.nbytes) // 32) * 32 == b.offset, a.name


def test_expert_sizes():
    sizes = [TABLE.expert_bytes(n) for n in range(numbers.LAYERS)]
    assert sizes[0] == 6_912_000
    assert sizes[25] == max(sizes) == 8_663_040
    assert sum(sizes[:20]) / 20 / 2**20 == pytest.approx(6.96, abs=0.01)
    assert sum(sizes[20:]) / 20 / 2**20 == pytest.approx(7.26, abs=0.01)
    assert all(t.nbytes % numbers.EXPERTS_PER_LAYER == 0 for t in TABLE.tensors if t.category is Category.EXPERTS)


def test_every_tensor_has_the_split_the_design_gives_it():
    expected = {
        "ffn_gate_exps.weight": SplitKind.EXPERT_ROWS,
        "ffn_up_exps.weight": SplitKind.EXPERT_ROWS,
        "ffn_down_exps.weight": SplitKind.EXPERT_COLUMNS,
        "ffn_gate_shexp.weight": SplitKind.ROWS,
        "ffn_up_shexp.weight": SplitKind.ROWS,
        "ffn_down_shexp.weight": SplitKind.COLUMNS,
        "attn_q_b.weight": SplitKind.ROWS,
        "attn_output_a.weight": SplitKind.ROWS,
        "attn_output_b.weight": SplitKind.COLUMNS,
        "engram_kv.weight": SplitKind.ROWS,
        "engram_gate.weight": SplitKind.ROWS,
    }
    outside = {
        "output.weight": SplitKind.ROWS,
        "output_norm.weight": SplitKind.WHOLE,
        "token_embd.weight": SplitKind.FILE,
        "per_layer_token_embd.weight": SplitKind.FILE,
    }
    assert len({t.name for t in TABLE.tensors}) == 1_043
    seen: set[str] = set()
    for t in TABLE.tensors:
        if t.layer < 0:
            assert split_of(t.name) is outside[t.name], t.name
        else:
            # Everything in a layer that the table above does not name is whole on both cards.
            short = t.name.split(".", 2)[2]
            seen.add(short)
            assert split_of(t.name) is expected.get(short, SplitKind.WHOLE), t.name
    assert seen >= expected.keys()
    assert "indexer.attn_q_b.weight" in seen
    with pytest.raises(KeyError):
        split_of("blk.0.a_tensor_with_no_rule.weight")


def test_card_bytes():
    """The indexer's `attn_q_b` is whole, as both cards score every position; the Engram gates are halved by stream."""
    card0, card1 = (sum(bytes_on_card(t, card) for t in TABLE.tensors) for card in (0, 1))
    assert (card0, card1) == (3_267_502_528, 3_267_502_528)
    indexer = NAMES["blk.2.indexer.attn_q_b.weight"]
    assert bytes_on_card(indexer, 0) == bytes_on_card(indexer, 1) == indexer.nbytes
    gate = NAMES["blk.1.engram_gate.weight"]
    assert bytes_on_card(gate, 0) == bytes_on_card(gate, 1) == gate.nbytes // 2


def test_an_experts_parts_are_exact_ninths():
    """Part A is 5/9 of an expert and part B 4/9, so a pair of slots takes exactly one expert's bytes on each card."""
    assert all(TABLE.expert_bytes(n) % 9 == 0 for n in range(numbers.LAYERS))


def test_the_unequal_shared_expert():
    """Q5_K in every layer: the larger part is on the card that does not post the misses, 5/9 and 4/9 of each matrix."""
    t = NAMES["blk.27.ffn_gate_shexp.weight"]
    assert t.type_name == "Q5_K"
    assert (bytes_on_card(t, 0), bytes_on_card(t, 1)) == (4_505_600, 3_604_480)
    t = NAMES["blk.28.ffn_down_shexp.weight"]
    assert (bytes_on_card(t, 0), bytes_on_card(t, 1)) == (3_604_480, 4_505_600)
    assert all(t.type_name == "Q5_K" for t in TABLE.tensors if "_shexp" in t.name)


def test_nothing_is_lost_or_doubled_by_a_split():
    for t in TABLE.tensors:
        on_cards = bytes_on_card(t, 0) + bytes_on_card(t, 1)
        match split_of(t.name):
            case SplitKind.WHOLE:
                assert on_cards == 2 * t.nbytes
            case SplitKind.ROWS | SplitKind.COLUMNS:
                assert on_cards == t.nbytes, t.name
            case _:
                assert on_cards == 0


def test_the_committed_file_is_what_to_tsv_writes(tmp_path: Path):
    written = tmp_path / "tensors.tsv"
    to_tsv(TABLE, written)
    assert written.read_bytes() == TSV.read_bytes()


def write_shards(first: Path) -> dict[str, np.ndarray]:
    """A GGUF in two shards, its tensors each with its own values."""
    arrays = {
        "output.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "blk.0.attn_kv.weight": np.arange(100, 110, dtype=np.float32),
        "blk.0.attn_norm.weight": np.arange(200, 206, dtype=np.float32).reshape(2, 3),
    }
    names = iter(arrays)
    for path, count in zip(shard_paths(first), (1, 2), strict=True):
        w = gguf.GGUFWriter(path, "deepseek41")
        for _ in range(count):
            name = next(names)
            w.add_tensor(name, arrays[name])
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    return arrays


def test_the_table_of_a_gguf_finds_each_tensors_bytes(tmp_path: Path):
    first = tmp_path / "m-00001-of-00002.gguf"
    arrays = write_shards(first)
    table = from_gguf(first)
    assert table.shard_paths == shard_paths(first)
    assert [(t.name, t.shard) for t in table.tensors] == [
        ("output.weight", 1),
        ("blk.0.attn_kv.weight", 2),
        ("blk.0.attn_norm.weight", 2),
    ]
    for t in table.tensors:
        assert (t.type_name, t.ggml_type) == ("F32", 0)
        assert t.shape == tuple(reversed(arrays[t.name].shape))  # ggml's order, the row length first
        data = table.shard_paths[t.shard - 1].read_bytes()[t.offset : t.offset + t.nbytes]
        assert data == arrays[t.name].tobytes(), t.name


def test_the_table_is_read_from_the_gguf_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    first = tmp_path / "m-00001-of-00002.gguf"
    write_shards(first)
    reads: list[Path] = []

    def counted(path: Path) -> tensors.TensorTable:
        reads.append(path)
        return from_gguf(path)

    monkeypatch.setattr(tensors, "from_gguf", counted)
    cache = tmp_path / "cache"
    a = from_gguf_cached(first, cache)
    b = from_gguf_cached(first, cache)
    assert reads == [first]
    assert b == a
    assert a.shard_paths == shard_paths(first)
