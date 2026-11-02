"""Download a model file and the parts that go with it into one directory (docs/GETTING_STARTED.md).

Fetches the file's shards, the image encoder and the DFlash 2 drafter at pinned revisions, and checks each shard's
size and sha256. A stopped download resumes when run again, and files already there are kept.

    python -m tools.model_fetch IQ3_S --dir /models
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tools import dflash_fetch

HUB = "https://huggingface.co"
CHUNK = 1 << 22
TRIES = 5
TIMEOUT = 60
"""Seconds without data before a read gives up and the download resumes."""
DRAFTER_SIZE = dflash_fetch.SIZE


@dataclass(frozen=True, slots=True)
class Source:
    repo: str
    revision: str


@dataclass(frozen=True, slots=True)
class RemoteFile:
    url: str
    name: str
    size: int
    sha256: str


ISTA = Source("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "ed59f92082b1e93c0e96d60a8b11aab089b52f09")
ENCODER = Source("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "1c04b8102ca5346f1faf4d9914503e378d713021")
UNSLOTH = Source("unsloth/Qwen3.8-Flash-Next-GGUF", "766911a6b7369840a91dbcd95f9f997acaab6cd6")
ENCODER_FILE = RemoteFile(
    f"{HUB}/{ENCODER.repo}/resolve/{ENCODER.revision}/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    907_543_008,
    "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0",
)
MODELS = {
    **dict.fromkeys(("IQ3_S", "IQ3_XXS", "IQ2_XS", "Q2_0"), ISTA),
    **dict.fromkeys(
        (
            "UD-IQ4_XS",
            "UD-IQ1_S",
            "UD-IQ1_M",
            "UD-Q2_K_XL",
            "UD-IQ3_XXS",
            "UD-Q3_K_XL",
            "UD-Q4_K_XL",
            "UD-Q5_K_XL",
            "UD-Q6_K_XL",
            "Q8_0",
        ),
        UNSLOTH,
    ),
}


def shards(name: str, hub: str = HUB) -> list[RemoteFile]:
    """The file's shards at the pinned revision. The Hub's listing at a commit cannot change, so its sums are pins."""
    source = MODELS[name]
    with urllib.request.urlopen(f"{hub}/api/models/{source.repo}/tree/{source.revision}/{name}") as r:
        listing = cast(list[dict[str, Any]], json.load(r))
    return [
        RemoteFile(
            f"{hub}/{source.repo}/resolve/{source.revision}/{item['path']}",
            Path(item["path"]).name,
            item["lfs"]["size"],
            item["lfs"]["oid"],
        )
        for item in sorted(listing, key=lambda i: i["path"])
        if item["path"].endswith(".gguf")
    ]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def link_shared(file: RemoteFile, directory: Path) -> None:
    """ISTA-DASLab's files share their second shard, so a copy already here saves its 28.8 GB."""
    dest = directory / file.name
    if dest.exists():
        return
    for other in directory.glob("*.gguf"):
        if other.stat().st_size == file.size and sha256_of(other) == file.sha256:
            dest.hardlink_to(other)
            print(f"{file.name}: linked to {other.name}", flush=True)
            return


def download(file: RemoteFile, directory: Path) -> None:
    """Fetch the file to a `.part` beside it, resuming what is there, and give it its name once it checks out."""
    dest = directory / file.name
    if dest.exists() and dest.stat().st_size == file.size:
        print(f"{file.name}: already here", flush=True)
        return
    part = directory / f"{file.name}.part"
    for attempt in range(1, TRIES + 1):
        try:
            _fetch_part(file, part)
            break
        except (urllib.error.URLError, OSError) as exc:
            if attempt == TRIES:
                raise SystemExit(f"{file.name}: {exc}. Run the same command again to resume.") from exc
            print(f"{file.name}: {exc}; trying again", flush=True)
            time.sleep(attempt)
    print(f"{file.name}: checking", flush=True)
    if part.stat().st_size != file.size or sha256_of(part) != file.sha256:
        part.unlink()
        raise SystemExit(f"{file.name}: size or sha256 does not match the pinned revision. Run the command again.")
    part.replace(dest)


def _fetch_part(file: RemoteFile, part: Path) -> None:
    have = part.stat().st_size if part.exists() else 0
    if have >= file.size:
        return
    request = urllib.request.Request(file.url, headers={"Range": f"bytes={have}-"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as r:
        # A server that ignores the range sends the whole file with 200.
        mode = "ab" if r.status == 206 else "wb"
        have = have if r.status == 206 else 0
        step = max(file.size // 20, 1)
        shown = have // step
        with part.open(mode) as f:
            while block := r.read(CHUNK):
                f.write(block)
                have += len(block)
                if have // step > shown:
                    shown = have // step
                    print(f"{file.name}: {have / 1e9:.1f} of {file.size / 1e9:.1f} GB", flush=True)


def missing_bytes(files: list[RemoteFile], directory: Path) -> int:
    total = 0
    for file in files:
        if not (directory / file.name).exists():
            part = directory / f"{file.name}.part"
            total += file.size - (part.stat().st_size if part.exists() else 0)
    return total


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m tools.model_fetch", description=(__doc__ or "").splitlines()[0])
    p.add_argument("model", choices=sorted(MODELS), help="the model file; docs/GETTING_STARTED.md compares them")
    p.add_argument("--dir", type=Path, default=Path("/models"), help="where the files go (default: /models)")
    a = p.parse_args(argv)
    directory: Path = a.dir
    directory.mkdir(parents=True, exist_ok=True)

    model = shards(a.model)
    files = [*model, ENCODER_FILE]
    for file in model:
        link_shared(file, directory)
    drafter = directory / "dflash"
    needs_drafter = not (drafter / "dflash-manifest.json").exists()
    need = missing_bytes(files, directory) + (DRAFTER_SIZE if needs_drafter else 0)
    free = shutil.disk_usage(directory).free
    if need > free:
        raise SystemExit(f"{directory} has {free / 1e9:.1f} GB free and this needs {need / 1e9:.1f} GB.")

    for file in files:
        download(file, directory)
    if needs_drafter:
        print("DFlash 2 drafter: downloading", flush=True)
        dflash_fetch.fetch(drafter)
    print(f"Done. Start the server with --model {directory / model[0].name}", flush=True)


if __name__ == "__main__":
    main()
