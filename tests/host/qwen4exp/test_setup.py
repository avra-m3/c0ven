from __future__ import annotations

from pathlib import Path

from engine.models.qwen4exp.setup import draft_dir


def test_draft_dir_prefers_dflash(tmp_path: Path):
    model = tmp_path / "model.gguf"
    assert draft_dir(model) is None
    (tmp_path / "mtp").mkdir()
    (tmp_path / "mtp" / "mtp-manifest.json").write_text("{}")
    assert draft_dir(model) == tmp_path / "mtp"
    (tmp_path / "dflash").mkdir()
    assert draft_dir(model) == tmp_path / "mtp"  # no manifest yet
    (tmp_path / "dflash" / "dflash-manifest.json").write_text("{}")
    assert draft_dir(model) == tmp_path / "dflash"
