"""Fetch the DFlash 2 drafter into `dflash/` beside the model files (MODEL_FILES.md, "The DFlash 2 drafter").

Downloads the drafter's one safetensors file and its config.json at a pinned revision, checks their sha256, and
writes each tensor the engine reads to `<out>/<tensor name>` as raw little-endian BF16 with no header. The candidate
selector's tensors are left out: the engine does not use them. The config's values the engine reads go into
`dflash-config.txt`, a line each: the key, then its values. `dflash-manifest.json` is written last, with the same
values; the engine finds the directory by it.

    python -m tools.dflash_fetch --out /models/dflash
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import urllib.request
from pathlib import Path
from typing import cast

REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
FILE = "model.safetensors"
SIZE = 1_206_140_008
SHA256 = "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"
SKIP = "candidate_selector."
CONFIG = "config.json"
CONFIG_SHA256 = "2409575986f9ea24d455343eb469c0e6940def5edab1d1be503bb8eb4cc3b1b9"
CONFIG_KEYS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "sliding_window",
    "rms_norm_eps",
)
"""The config's top-level values the engine reads; dflash_config's and rope_parameters' are all read."""
CHUNK = 1 << 22


def config_values() -> dict[str, object]:
    """The values of the drafter's config.json the engine reads, flattened: dflash_config's (the mask token, the
    target layers, the convolutions and the block) and rope_parameters' rope_theta beside the sizes."""
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{CONFIG}"
    with urllib.request.urlopen(url) as r:
        body = r.read()
    if hashlib.sha256(body).hexdigest() != CONFIG_SHA256:
        raise SystemExit(f"{CONFIG}: sha256 does not match the pinned revision")
    config = json.loads(body)
    values: dict[str, object] = {k: config[k] for k in CONFIG_KEYS}
    values["rope_theta"] = config["rope_parameters"]["rope_theta"]
    values.update(config["dflash_config"])
    return values


def config_text(values: dict[str, object]) -> str:
    """dflash-config.txt: a line for each value, the key then its numbers."""
    lines: list[str] = []
    for key, v in values.items():
        numbers = cast("list[object]", v) if isinstance(v, list) else [v]
        lines.append(" ".join([key, *(str(n) for n in numbers)]))
    return "\n".join(lines) + "\n"


def download(dest: Path) -> None:
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILE}"
    digest = hashlib.sha256()
    with urllib.request.urlopen(url) as r, dest.open("wb") as f:
        while block := r.read(CHUNK):
            digest.update(block)
            f.write(block)
    if dest.stat().st_size != SIZE or digest.hexdigest() != SHA256:
        raise SystemExit(f"{FILE}: size or sha256 does not match the pinned revision")


def split(src: Path, out: Path) -> list[dict[str, object]]:
    tensors: list[dict[str, object]] = []
    with src.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        for name, t in sorted(header.items()):
            if name == "__metadata__" or name.startswith(SKIP):
                continue
            if t["dtype"] != "BF16":
                raise SystemExit(f"{name}: {t['dtype']}, not BF16")
            begin, end = t["data_offsets"]
            f.seek(8 + n + begin)
            data = f.read(end - begin)
            (out / name).write_bytes(data)
            tensors.append(
                {
                    "name": name,
                    "dtype": "BF16",
                    "shape": t["shape"],
                    "range": [8 + n + begin, 8 + n + end],
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    return tensors


def fetch(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "dflash-manifest.json"
    manifest.unlink(missing_ok=True)
    tmp = out / f".{FILE}.{os.getpid()}"
    try:
        download(tmp)
        tensors = split(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    config = config_values()
    (out / "dflash-config.txt").write_text(config_text(config))
    body = {"repo": REPO, "revision": REVISION, "file": FILE, "sha256": SHA256, "config": config, "tensors": tensors}
    manifest.write_text(json.dumps(body, indent=1) + "\n")
    print(f"{len(tensors)} tensors in {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch the DFlash 2 drafter into dflash/ beside the model files.")
    p.add_argument("--out", type=Path, required=True, help="the directory to write, normally dflash/ beside the model")
    fetch(p.parse_args().out)


if __name__ == "__main__":
    main()
