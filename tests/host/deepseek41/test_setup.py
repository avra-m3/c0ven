from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from engine.api.interface import SplitKind, Tensor
from engine.memory_plan import GIB, PlanError, PlanOptions, ngram_fits
from engine.models import ModelSetup
from engine.models.deepseek41 import setup
from engine.models.deepseek41.tensors import TensorTable, from_tsv
from engine.process import model_module

TABLE = from_tsv()


def test_the_setup_module_is_a_model_setup():
    found: ModelSetup = setup  # basedpyright checks the module against the Protocol
    assert model_module("deepseek41") is found


def test_no_drafter_and_no_images_even_beside_the_files(tmp_path: Path):
    model = tmp_path / "model-00001-of-00006.gguf"
    (tmp_path / "mtp").mkdir()
    (tmp_path / "mtp" / "mtp-manifest.json").write_text("{}")
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"")
    assert setup.draft_dir(model) is None
    assert setup.vision_file(model) is None


def test_the_open_config_without_a_model():
    options = PlanOptions(slots=2, context=262_144, checkpoints=3, host_memory=64 * GIB)
    config, plan = setup.open_config(None, options, deterministic=True, cpu_workers=7)
    assert config.tensors == ()
    assert config.max_context == 16_384
    assert config.slots == 2
    assert config.pages == plan.pages == 2 * 4_096
    assert config.checkpoints == 3
    assert config.deterministic
    assert config.cpu_workers == 7
    assert config.kv_resident_cells == 16_384
    assert config.experts_cached == plan.experts_cached
    assert config.arena_bytes == plan.arena_bytes > 0
    assert config.working_bytes == plan.working_per_card
    assert config.eos_ids == (1,)
    assert config.vocab_size == 129_280
    assert config.ngram_cache_rows == 1 << 20
    assert config.draft_dir == ""
    assert not config.ngram_in_ram


def test_a_model_file_stops_with_the_reason_while_the_core_lacks_the_model(tmp_path: Path):
    with pytest.raises(PlanError, match="core does not run deepseek41"):
        setup.open_config(tmp_path / "m-00001-of-00006.gguf", PlanOptions())


def test_the_open_config_passes_each_tensor_of_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    paths = tuple(tmp_path / f"m-0000{i}-of-00006.gguf" for i in range(1, 7))

    def read(model: Path, cache: Path | None) -> TensorTable:
        assert (model, cache) == (paths[0], tmp_path)
        return replace(TABLE, shard_paths=paths)

    monkeypatch.setattr(setup, "CORE_HAS_MODEL", True)
    monkeypatch.setattr(setup, "from_gguf_cached", read)
    monkeypatch.setenv("ENGINE_CPU_WORKERS", "5")
    config, _ = setup.open_config(paths[0], PlanOptions(host_memory=64 * GIB), table_cache=tmp_path, cpu_workers=7)
    assert config.shard_paths == tuple(str(p) for p in paths)
    assert config.cpu_workers == 5
    by_name = {t.name: t for t in config.tensors}
    assert len(by_name) == 1_043
    # Shards count from 0 here and from 1 in the file names; the offsets are the table's own.
    assert by_name["output.weight"] == Tensor(
        "output.weight", 13, 1, 22_112, 455_065_600, (5_120, 129_280), SplitKind.ROWS, -1
    )
    expert = by_name["blk.14.ffn_down_exps.weight"]
    assert (expert.shard, expert.offset, expert.split, expert.layer) == (2, 3_001_837_184, SplitKind.EXPERT_COLUMNS, 14)


def test_the_engram_table_stays_on_the_ssd_when_the_host_has_room():
    """The engine process asks for the table in RAM by itself when `ngram_fits` says there is room."""
    roomy = PlanOptions(host_memory=384 * GIB)
    config, plan = setup.open_config(None, roomy)
    assert ngram_fits(plan, 384 * GIB)
    asked_config, asked_plan = setup.open_config(None, replace(roomy, ngram_in_ram=True))
    assert not asked_config.ngram_in_ram
    assert not asked_plan.options.ngram_in_ram
    assert asked_plan == plan
    assert asked_config == config
