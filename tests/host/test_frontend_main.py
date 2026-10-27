"""The front end's command line, and the engine's that it builds."""

from __future__ import annotations

import contextlib
import re
import tempfile
from pathlib import Path

import pytest

from frontend.__main__ import engine_argv, parse_args, table_cache, version


def test_version_prints_the_projects_version(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit) as e:
        parse_args(["--version"])
    assert e.value.code == 0
    assert capsys.readouterr().out == f"c0ven {version()}\n"
    assert re.fullmatch(r"\d+\.\d+\.\d+", version())


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


def test_the_kv_pool_reaches_the_engine_only_when_set():
    args = parse_args(["--engine-backend", "mock", "--slots", "4", "--kv-tokens", "512k"])
    argv = engine_argv(args, "", None)
    assert argv[argv.index("--kv-tokens") + 1] == "524288"
    assert "--kv-tokens" not in engine_argv(parse_args(["--engine-backend", "mock"]), "", None)


def test_the_default_limit_takes_a_token_count():
    assert parse_args(["--default-max-tokens", "32k"]).default_max_tokens == 32_768
    assert parse_args([]).default_max_tokens == 0


def test_metrics_are_not_logged_unless_asked():
    assert parse_args([]).metrics_log == 0
    assert parse_args(["--metrics-log", "60"]).metrics_log == 60
