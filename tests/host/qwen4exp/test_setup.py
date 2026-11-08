from __future__ import annotations

from pathlib import Path

from engine.models import ModelSetup
from engine.models.qwen4exp import setup
from engine.models.qwen4exp.setup import draft_dir, vision_file
from engine.process import model_module


def test_the_setup_module_is_a_model_setup():
    found: ModelSetup = setup  # basedpyright checks the module against the Protocol
    assert model_module("qwen4exp") is found


def test_draft_dir_prefers_dflash(tmp_path: Path):
    model = tmp_path / "model.gguf"
    assert draft_dir(model) is None
    (tmp_path / "mtp").mkdir()
    (tmp_path / "mtp" / "mtp-manifest.json").write_text("{}")
    assert draft_dir(model) == tmp_path / "mtp"
    (tmp_path / "dflash").mkdir()
    (tmp_path / "dflash" / "model.safetensors").write_bytes(b"")
    assert draft_dir(model) == tmp_path / "mtp"  # no config.json yet
    (tmp_path / "dflash" / "config.json").write_text("{}")
    assert draft_dir(model) == tmp_path / "dflash"


def test_vision_file_is_the_one_mmproj_beside_the_model(tmp_path: Path):
    model = tmp_path / "model-00001-of-00002.gguf"
    assert vision_file(model) is None
    (tmp_path / "mmproj-BF16.gguf").write_bytes(b"")
    assert vision_file(model) == tmp_path / "mmproj-BF16.gguf"
    (tmp_path / "mmproj-F16.gguf").write_bytes(b"")
    assert vision_file(model) is None
