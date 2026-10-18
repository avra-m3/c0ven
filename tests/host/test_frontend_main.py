"""The front end's command line, and the engine's that it builds."""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path

import pytest

from frontend.__main__ import engine_argv, parse_args, table_cache


def test_the_native_engine_gets_the_architecture_and_the_table_cache():
    """So the engine reads the GGUF's tensor table once per front end, and its header never."""
    args = parse_args(["--model", "/models/m-00001-of-00002.gguf", "--slots", "2"])
    argv = engine_argv(args, "qwen4exp", Path("/tmp/cache"))
    assert argv[argv.index("--model") + 1] == "/models/m-00001-of-00002.gguf"
    assert argv[argv.index("--arch") + 1] == "qwen4exp"
    assert argv[argv.index("--table-cache") + 1] == "/tmp/cache"


def test_the_mock_engine_has_no_model_file():
    args = parse_args(["--engine-backend", "mock", "--arch", "qwen4exp"])
    argv = engine_argv(args, "", Path("/tmp/cache"))
    assert argv[argv.index("--arch") + 1] == "qwen4exp"
    assert "--model" not in argv
    assert "--table-cache" not in argv


def test_without_a_writable_directory_the_engine_reads_its_own_table(monkeypatch: pytest.MonkeyPatch):
    """A container with a read-only root has no temporary directory: the front end still starts."""

    def nowhere(*_: object, **__: object) -> tempfile.TemporaryDirectory[str]:
        raise FileNotFoundError("No usable temporary directory found")

    monkeypatch.setattr(tempfile, "TemporaryDirectory", nowhere)
    with contextlib.ExitStack() as stack:
        cache = table_cache(stack)
    assert cache is None
    args = parse_args(["--model", "/models/m-00001-of-00002.gguf"])
    assert "--table-cache" not in engine_argv(args, "qwen4exp", cache)
