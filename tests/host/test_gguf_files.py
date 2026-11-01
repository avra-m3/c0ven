"""A split GGUF's file names."""

from __future__ import annotations

from pathlib import Path

from engine.gguf_files import first_shard, shard_paths


def test_any_shard_names_the_first():
    assert first_shard(Path("/m/x-00004-of-00006.gguf")) == Path("/m/x-00001-of-00006.gguf")
    assert first_shard(Path("/m/x-00001-of-00006.gguf")) == Path("/m/x-00001-of-00006.gguf")
    assert first_shard(Path("/m/one.gguf")) == Path("/m/one.gguf")


def test_any_shard_names_them_all():
    shard = Path("/m/x-00002-of-00002.gguf")
    assert [p.name for p in shard_paths(shard)] == [
        "x-00001-of-00002.gguf",
        "x-00002-of-00002.gguf",
    ]
    assert shard_paths(Path("/m/one.gguf")) == (Path("/m/one.gguf"),)
