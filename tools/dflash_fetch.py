"""Fetch the DFlash 2 drafter into `dflash/` beside the model files (MODEL_FILES.md, "The DFlash 2 drafter").

Downloads the drafter's one safetensors file at a pinned revision, checks its sha256, and writes each tensor the
engine reads to `<out>/<tensor name>` as raw little-endian BF16 with no header. The candidate selector's tensors are
left out: the engine does not use them. `dflash-manifest.json` is written last; the engine finds the directory by it.

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

REPO = "incoai/Qwen3.8-Flash-Next-DFlash2"
REVISION = "a22a1744e85c17364dcc54a6a420af7135d4c7f6"
FILE = "model.safetensors"
SIZE = 1_206_140_008
SHA256 = "d6723f2a359eaf33ec0c95f22ae8edda896774020c552296a85e1913b4d3186f"
SKIP = "candidate_selector."
CHUNK = 1 << 22


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


def main() -> None:
    p = argparse.ArgumentParser(description="Fetch the DFlash 2 drafter into dflash/ beside the model files.")
    p.add_argument("--out", type=Path, required=True, help="the directory to write, normally dflash/ beside the model")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out / "dflash-manifest.json"
    manifest.unlink(missing_ok=True)
    tmp = a.out / f".{FILE}.{os.getpid()}"
    try:
        download(tmp)
        tensors = split(tmp, a.out)
    finally:
        tmp.unlink(missing_ok=True)
    body = {"repo": REPO, "revision": REVISION, "file": FILE, "sha256": SHA256, "tensors": tensors}
    manifest.write_text(json.dumps(body, indent=1) + "\n")
    print(f"{len(tensors)} tensors in {a.out}")


if __name__ == "__main__":
    main()
