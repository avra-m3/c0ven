"""The disk tier: checkpoints kept in files, so a checkpoint the KV pool or the checkpoint pool has no room for is
not lost (docs/CACHING.md, "The disk tier"; ADR-0082).

An entry is one checkpoint: a state file (the core's fixed state and the tokens), the KV of its whole blocks of
BLOCK tokens in block files, and a tail file of the pages after its last whole block. A block file is named by the
hash of every token up to the block's end, so the checkpoints of one conversation share the block files of the
prefix they share, and a block is written once. The running state cannot be wound back, so an entry is restored at
its own length only, never at a prefix of it.

The files are under root/<identity>, the identity naming the model file, the engine's version and the cards, so no
load reads a file another wrote. Entries outlive the process: at open, the folder's entries are read back, and when
the folder passes its cap the entries used longest ago go first. A file the core refuses or cannot read takes its
entry with it. One server uses a folder at a time.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from engine.api.interface import PAGE_CELLS, Engine, EngineError

log = logging.getLogger("engine.seqstate")

BLOCK = 1024
"""Tokens a block file holds: 256 pages."""
BLOCK_PAGES = BLOCK // PAGE_CELLS
MIN_TOKENS = 1024
"""The shortest checkpoint kept on disk. A shorter prefix reads again in about the time its files take."""

type Chain = list[bytes]
"""The hashes of a prompt's whole blocks: chain[b] stands for its tokens 0 .. (b + 1) * BLOCK - 1."""


def chain_of(tokens: Sequence[int], blocks: int | None = None) -> Chain:
    """The hashes of the first `blocks` whole blocks of `tokens`, all of them by default."""
    n = len(tokens) // BLOCK if blocks is None else min(blocks, len(tokens) // BLOCK)
    ids = np.asarray(tokens[: n * BLOCK], dtype=np.int64).astype("<i4")
    out: Chain = []
    h = b""
    for b in range(n):
        h = hashlib.blake2b(h + ids[b * BLOCK : (b + 1) * BLOCK].tobytes(), digest_size=16).digest()
        out.append(h)
    return out


def entry_name(tokens: Sequence[int]) -> str:
    return hashlib.blake2b(np.asarray(tokens, dtype=np.int64).astype("<i4").tobytes(), digest_size=16).hexdigest()


def storable(tokens: Sequence[int]) -> bool:
    """Whether a checkpoint of these tokens can go to disk: long enough, and no image (an image's tokens are negative
    in a prefix key, and the core keeps no pixels in a file)."""
    return len(tokens) >= MIN_TOKENS and min(tokens) >= 0


@dataclass(slots=True)
class Entry:
    name: str
    length: int
    blocks: tuple[bytes, ...]
    tail: tuple[int, ...]
    """The tokens after the last whole block."""
    tail_pages: int
    bytes: int
    """Its own files' bytes: the state file, the tail file and this record."""
    used: float


@dataclass(frozen=True, slots=True)
class DiskStats:
    entries: int = 0
    bytes: int = 0
    saved: int = 0
    """Checkpoints written to disk since open."""
    loaded: int = 0
    """Checkpoints read back."""
    save_s: float = 0.0
    load_s: float = 0.0
    evicted: int = 0
    """Entries removed for the cap."""


class DiskTier:
    def __init__(self, root: Path, identity: str, cap_bytes: int) -> None:
        self.dir = root / identity
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cap = cap_bytes
        self._entries: dict[str, Entry] = {}
        self._blocks: dict[bytes, int] = {}
        """Each block file's bytes."""
        self._refs: dict[bytes, int] = {}
        """Entries using each block file."""
        self._saved = self._loaded = self._evicted = 0
        self._save_s = self._load_s = 0.0
        self._scan()

    # Files.

    def _block(self, h: bytes) -> Path:
        return self.dir / f"b-{h.hex()}.kv"

    def _state(self, name: str) -> Path:
        return self.dir / f"c-{name}.state"

    def _tail(self, name: str) -> Path:
        return self.dir / f"c-{name}.tail"

    def _record(self, name: str) -> Path:
        return self.dir / f"c-{name}.json"

    def _scan(self) -> None:
        """Read back the folder's entries; remove a file no whole entry uses, and an entry missing a file."""
        for record in self.dir.glob("c-*.json"):
            name = record.stem.removeprefix("c-")
            try:
                d = json.loads(record.read_text())
                blocks = tuple(bytes.fromhex(h) for h in d["blocks"])
                entry = Entry(name, int(d["length"]), blocks, tuple(d["tail"]), int(d["tail_pages"]), 0, 0.0)
                files = [self._state(name), record] + ([self._tail(name)] if entry.tail_pages else [])
                entry.bytes = sum(f.stat().st_size for f in files)
                entry.used = record.stat().st_mtime
                for h in blocks:
                    self._blocks.setdefault(h, self._block(h).stat().st_size)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                log.info("disk tier: entry %s is incomplete and removed: %s", name, exc)
                self._unlink_entry(name)
                continue
            self._entries[name] = entry
            for h in blocks:
                self._refs[h] = self._refs.get(h, 0) + 1
        used = {f"b-{h.hex()}.kv" for h in self._refs}
        for f in self.dir.iterdir():
            if f.name.endswith(".tmp") or not (f.name in used or f.stem.removeprefix("c-") in self._entries):
                f.unlink(missing_ok=True)
        self._blocks = {h: b for h, b in self._blocks.items() if h in self._refs}
        if self._entries:
            log.info("disk tier: %d checkpoints, %.1f GiB, in %s", len(self._entries), self.total / 2**30, self.dir)

    def _unlink_entry(self, name: str) -> None:
        for f in (self._state(name), self._tail(name), self._record(name)):
            f.unlink(missing_ok=True)

    @property
    def total(self) -> int:
        return sum(e.bytes for e in self._entries.values()) + sum(self._blocks.values())

    def stats(self) -> DiskStats:
        return DiskStats(
            len(self._entries), self.total, self._saved, self._loaded, self._save_s, self._load_s, self._evicted
        )

    def __len__(self) -> int:
        return len(self._entries)

    # Lookup.

    def find(self, tokens: Sequence[int]) -> Entry | None:
        return self._entries.get(entry_name(tokens)) if storable(tokens) else None

    def longest_prefix(self, key: tuple[int, ...], chain: Chain) -> Entry | None:
        """The longest entry whose tokens are a strict prefix of `key`; `chain` is chain_of(key)."""
        best: Entry | None = None
        for e in self._entries.values():
            nb = len(e.blocks)
            if e.length >= len(key) or (best is not None and e.length <= best.length) or nb > len(chain):
                continue
            if (nb == 0 or chain[nb - 1] == e.blocks[-1]) and key[nb * BLOCK : e.length] == e.tail:
                best = e
        return best

    def touch(self, entry: Entry) -> None:
        entry.used = time.time()
        with contextlib.suppress(OSError):
            os.utime(self._record(entry.name))

    # Writes and reads.

    def save(self, engine: Engine, checkpoint_id: int, tokens: tuple[int, ...], pages: Sequence[int]) -> bool:
        """Write a checkpoint the core has answered, of `tokens` and holding `pages` (logical order), unless its
        entry is there already or it cannot go to disk. Then entries used longest ago go until the folder is within
        its cap. False if it is not on disk after."""
        if not storable(tokens):
            return False
        name = entry_name(tokens)
        if (existing := self._entries.get(name)) is not None:
            self.touch(existing)
            return True
        t0 = time.monotonic()
        chain = chain_of(tokens)
        nb = len(chain)
        tail_pages = len(pages) - nb * BLOCK_PAGES
        written: list[Path] = []
        try:
            for b, h in enumerate(chain):
                if h in self._blocks:
                    continue
                block = pages[b * BLOCK_PAGES : (b + 1) * BLOCK_PAGES]
                self._write(lambda path, block=block: engine.save_pages(block, path), self._block(h))
                written.append(self._block(h))
                self._blocks[h] = self._block(h).stat().st_size
            if tail_pages:
                tail = pages[nb * BLOCK_PAGES :]
                self._write(lambda path: engine.save_pages(tail, path), self._tail(name))
            self._write(lambda path: engine.save_checkpoint(checkpoint_id, path), self._state(name))
            record = {"length": len(tokens), "blocks": [h.hex() for h in chain], "tail": list(tokens[nb * BLOCK :])}
            tmp = self._record(name).with_suffix(".json.tmp")
            tmp.write_text(json.dumps({**record, "tail_pages": tail_pages}))
            tmp.replace(self._record(name))
        except (EngineError, OSError) as exc:
            log.warning("disk tier: checkpoint of %d tokens not written: %s", len(tokens), exc)
            self._unlink_entry(name)
            for f in written:
                f.unlink(missing_ok=True)
            self._blocks = {h: b for h, b in self._blocks.items() if h in self._refs}
            return False
        files = [self._state(name), self._record(name)] + ([self._tail(name)] if tail_pages else [])
        entry = Entry(name, len(tokens), tuple(chain), tuple(tokens[nb * BLOCK :]), tail_pages, 0, time.time())
        entry.bytes = sum(f.stat().st_size for f in files)
        self._entries[name] = entry
        for h in chain:
            self._refs[h] = self._refs.get(h, 0) + 1
        self._saved += 1
        self._save_s += time.monotonic() - t0
        log.info(
            "disk tier: checkpoint of %d tokens written in %.2f s, %d of its %d blocks new",
            len(tokens),
            time.monotonic() - t0,
            len(written),
            nb,
        )
        self._fit(keep=name)
        return name in self._entries

    @staticmethod
    def _write(save: Callable[[Path], None], path: Path) -> None:
        """`save` to a temporary name, then rename it: a file is whole or absent."""
        tmp = path.with_name(path.name + ".tmp")
        save(tmp)
        tmp.replace(path)

    def load(self, engine: Engine, entry: Entry, checkpoint_id: int, pages: Sequence[int]) -> bool:
        """Read an entry into a checkpoint and its pages (logical order, as many as its positions take). False, with
        the entry removed, if a file is missing or refused."""
        t0 = time.monotonic()
        nb = len(entry.blocks)
        try:
            for b, h in enumerate(entry.blocks):
                engine.load_pages(pages[b * BLOCK_PAGES : (b + 1) * BLOCK_PAGES], self._block(h))
            if entry.tail_pages:
                engine.load_pages(pages[nb * BLOCK_PAGES :], self._tail(entry.name))
            engine.load_checkpoint(checkpoint_id, self._state(entry.name), entry.length)
        except EngineError as exc:
            log.warning("disk tier: checkpoint of %d tokens not read, and removed: %s", entry.length, exc)
            self.remove(entry)
            return False
        self.touch(entry)
        self._loaded += 1
        self._load_s += time.monotonic() - t0
        log.info("disk tier: checkpoint of %d tokens read in %.2f s", entry.length, time.monotonic() - t0)
        return True

    def remove(self, entry: Entry) -> None:
        self._entries.pop(entry.name, None)
        self._unlink_entry(entry.name)
        for h in entry.blocks:
            self._refs[h] -= 1
            if self._refs[h] == 0:
                del self._refs[h]
                self._blocks.pop(h, None)
                self._block(h).unlink(missing_ok=True)

    def _fit(self, keep: str) -> None:
        """Remove the entries used longest ago until the folder is within its cap; `keep` last of all."""
        while self.total > self.cap and self._entries:
            victim = min(self._entries.values(), key=lambda e: (e.name == keep, e.used))
            log.info("disk tier: checkpoint of %d tokens removed: the folder is over its cap", victim.length)
            self.remove(victim)
            self._evicted += 1
