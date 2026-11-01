"""tools/plan.py: the plan printed, for the architecture of the model file or the committed table's."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers import write_gguf
from tools.plan import main


def test_without_a_model_the_committed_table_is_planned(capsys: pytest.CaptureFixture[str]):
    assert main(["--slots", "1", "--context", "32768"]) == 0
    assert "experts cached" in capsys.readouterr().out


def test_the_architecture_is_the_model_files(tmp_path: Path):
    write_gguf(tmp_path / "m-00001-of-00002.gguf", "testarch")
    with pytest.raises(SystemExit, match="'testarch' is not supported"):
        main(["--model", str(tmp_path / "m-00002-of-00002.gguf")])


def test_arch_names_the_architecture_without_a_model():
    with pytest.raises(SystemExit, match="'testarch' is not supported"):
        main(["--arch", "testarch"])
