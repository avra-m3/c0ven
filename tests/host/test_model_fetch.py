"""tools/model_fetch.py: resuming, checking and sharing files, against a local server that honours ranges."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tools import model_fetch
from tools.model_fetch import RemoteFile

BODY = bytes(range(256)) * 4096


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        begin = 0
        if value := self.headers.get("Range"):
            begin = int(value.removeprefix("bytes=").removesuffix("-"))
        self.send_response(206 if begin else 200)
        self.end_headers()
        self.wfile.write(BODY[begin:])

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}/file"
    server.shutdown()


def remote(url: str, sha256: str | None = None) -> RemoteFile:
    return RemoteFile(url, "m.gguf", len(BODY), sha256 or hashlib.sha256(BODY).hexdigest())


def test_download_from_nothing(url: str, tmp_path: Path) -> None:
    model_fetch.download(remote(url), tmp_path)
    assert (tmp_path / "m.gguf").read_bytes() == BODY
    assert not (tmp_path / "m.gguf.part").exists()


def test_download_resumes_a_part(url: str, tmp_path: Path) -> None:
    (tmp_path / "m.gguf.part").write_bytes(BODY[:1000])
    model_fetch.download(remote(url), tmp_path)
    assert (tmp_path / "m.gguf").read_bytes() == BODY


def test_download_refuses_a_wrong_sum(url: str, tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="sha256"):
        model_fetch.download(remote(url, "0" * 64), tmp_path)
    assert not (tmp_path / "m.gguf").exists()
    assert not (tmp_path / "m.gguf.part").exists()


def test_download_keeps_a_file_already_there(tmp_path: Path) -> None:
    (tmp_path / "m.gguf").write_bytes(BODY)
    model_fetch.download(remote("http://127.0.0.1:1/unused"), tmp_path)
    assert (tmp_path / "m.gguf").read_bytes() == BODY


def test_a_shared_shard_is_linked(tmp_path: Path) -> None:
    (tmp_path / "other.gguf").write_bytes(BODY)
    model_fetch.link_shared(remote("http://127.0.0.1:1/unused"), tmp_path)
    assert (tmp_path / "m.gguf").samefile(tmp_path / "other.gguf")


def test_a_different_file_of_the_same_size_is_not_linked(tmp_path: Path) -> None:
    (tmp_path / "other.gguf").write_bytes(bytes(len(BODY)))
    model_fetch.link_shared(remote("http://127.0.0.1:1/unused"), tmp_path)
    assert not (tmp_path / "m.gguf").exists()


def test_the_start_command_printed_is_the_readmes() -> None:
    shard = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf"
    printed = model_fetch.finished(model_fetch.DEFAULT_DIR, shard).splitlines()[-1]
    assert printed in Path("README.md").read_text()


def test_another_directory_gets_only_the_model_path(tmp_path: Path) -> None:
    assert model_fetch.finished(tmp_path, "m.gguf") == f"Done. Start the server with --model {tmp_path / 'm.gguf'}"


def test_missing_bytes_counts_what_is_left(tmp_path: Path) -> None:
    (tmp_path / "a.gguf").write_bytes(b"x")
    (tmp_path / "b.gguf.part").write_bytes(b"xxx")
    files = [RemoteFile("", "a.gguf", 1, ""), RemoteFile("", "b.gguf", 10, ""), RemoteFile("", "c.gguf", 5, "")]
    assert model_fetch.missing_bytes(files, tmp_path) == 12
