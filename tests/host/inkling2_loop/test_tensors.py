from __future__ import annotations

from pathlib import Path

import pytest

from engine.api.interface import SplitKind
from engine.models.inkling2_loop import tensors
from engine.models.inkling2_loop.split import split_of


def test_census_metadata_and_token_ids():
    table = tensors.from_tsv()
    assert table.architecture == "inkling2_loop"
    assert table.layers == 24
    assert table.logical_layers == 48
    assert table.swa_layers == 40
    assert table.global_layers == 8
    assert table.head_count == 24
    assert table.head_count_kv == 8
    assert table.head_dim == 128
    assert table.experts_per_layer == 128
    assert table.experts_used == 8
    assert table.expert_ff == 1152
    assert table.vocab_size == 201_024
    assert table.vocab_unpadded == 200_058
    assert table.token_id("<|message_user|>") == 200_000
    assert table.token_id("<|message_model|>") == 200_001
    assert table.token_id("<|end_message|>") == 200_010
    assert table.token_id("<|content_model_end_sampling|>") == 200_006
    assert table.token_id("<|unused_200054|>") == 200_054


def test_split_of_all_tensor_kinds():
    table = tensors.from_tsv()
    kinds = {t.name: split_of(t) for t in table.tensors}
    assert kinds["token_embd.weight"] is SplitKind.FILE
    assert kinds["token_embd_norm.weight"] is SplitKind.WHOLE
    assert kinds["output.weight"] is SplitKind.ROWS
    assert kinds["output_norm.weight"] is SplitKind.WHOLE
    assert kinds["blk.0.attn_q.weight"] is SplitKind.ROWS
    assert kinds["blk.0.attn_k.weight"] is SplitKind.ROWS
    assert kinds["blk.0.attn_v.weight"] is SplitKind.ROWS
    assert kinds["blk.0.attn_output.weight"] is SplitKind.COLUMNS
    assert kinds["blk.0.attn_rel_bias.weight"] is SplitKind.WHOLE
    assert kinds["blk.6.loop_q_ctrl_down.weight"] is SplitKind.WHOLE
    assert kinds["blk.0.ffn_gate_inp.weight"] is SplitKind.WHOLE
    assert kinds["blk.0.ffn_gate_up_exps.weight"] is SplitKind.EXPERT_ROWS
    assert kinds["blk.0.ffn_down_exps.weight"] is SplitKind.EXPERT_COLUMNS


def test_table_cache_reads_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"v1")
    reads: list[Path] = []

    def fake_from_gguf(path: Path) -> tensors.TensorTable:
        reads.append(path)
        base = tensors.from_tsv()
        return tensors.TensorTable(base.tensors, (path,), base.fields)

    monkeypatch.setattr(tensors, "from_gguf", fake_from_gguf)
    cache = tmp_path / "cache"
    a = tensors.from_gguf_cached(model, cache)
    b = tensors.from_gguf_cached(model, cache)
    assert reads == [model]
    assert a == b
    model.write_bytes(b"changed")
    tensors.from_gguf_cached(model, cache)
    assert reads == [model, model]
