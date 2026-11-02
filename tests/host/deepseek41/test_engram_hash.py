"""deepseek41's Engram hash, ported from DeepSeek's inference/engram.py (NgramHashState, and the layout and multipliers
it is built from), against reference/deepseek41/engram_rows.json: the file's hash numbers, a subset of its token map
and 1,000 windows with their rows. core/tests/ds41_engram_test.cpp checks the C++ hash against the same rows.

To write the JSON again from the model's first shard: python tests/host/deepseek41/test_engram_hash.py <shard 1>.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

ROWS = Path(__file__).resolve().parents[3] / "reference" / "deepseek41" / "engram_rows.json"
TABLE_ROWS = 768_022_850  # per_layer_token_embd's rows in the file
DEAD = -1  # NgramHashState.DEAD: a token that takes no part in an n-gram (an image token)
IMAGE_TOKEN = 129_264  # inference/config.json's image_token_id: an image's span is made of it
WINDOWS = 1000


@dataclass(frozen=True)
class Config:
    """The hash's settings, as the file's metadata holds them."""

    layers: tuple[int, ...]  # ple.layers
    max_ngram: int  # ple.ngram_size
    heads: int  # ple.heads_per_ngram
    bucket_start: int  # engram.vocab_size
    compressed_vocab: int  # engram.compressed_vocab_size
    pad: int  # engram.pad_token_id


@dataclass(frozen=True)
class Layout:
    """engram.py's EngramLayout and multipliers: per table, the prime of each (n-gram, head) column, each column's
    first row within its table, and the multiplier of each look-back."""

    primes: tuple[tuple[int, ...], ...]  # [table][(n - 2) * heads + head]
    offsets: tuple[tuple[int, ...], ...]  # [table][column], from 0 in each table
    multipliers: tuple[tuple[int, ...], ...]  # [table][shift]

    def bases(self) -> list[int]:
        """Each table's first row in the file, which holds the tables one after the other."""
        return [sum(sum(p) for p in self.primes[:e]) for e in range(len(self.primes))]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def find_next_prime(start: int, seen: set[int]) -> int:
    """engram.py find_next_prime: the smallest prime above start not yet handed out."""
    candidate = start + 1
    while not is_prime(candidate) or candidate in seen:
        candidate += 1
    return candidate


def layout(config: Config) -> Layout:
    """engram.py EngramLayout.from_args, the offsets of NgramHashState.__init__ and compute_hash_multipliers."""
    primes: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for _ in config.layers:
        flat: list[int] = []
        for _ in range(config.max_ngram - 1):
            current = config.bucket_start - 1
            for _ in range(config.heads):
                current = find_next_prime(current, seen)
                seen.add(current)
                flat.append(current)
        primes.append(tuple(flat))
    offsets = [tuple(sum(p[:i]) for i in range(len(p))) for p in primes]
    bound = max(1, (np.iinfo(np.int64).max // config.compressed_vocab) // 2)
    multipliers: list[tuple[int, ...]] = []
    for layer in config.layers:
        values = np.random.default_rng(10007 * layer).integers(
            low=0, high=bound, size=(config.max_ngram,), dtype=np.int64
        )
        multipliers.append(tuple(int(v) * 2 + 1 for v in values))
    return Layout(tuple(primes), tuple(offsets), tuple(multipliers))


def ngram_hashes(
    input_ids: list[int],
    start_pos: int,
    token_mask: list[bool],
    cache: dict[int, int],
    *,
    token_map: dict[int, int],
    config: Config,
    lay: Layout,
) -> list[list[list[int]]]:
    """NgramHashState.forward for one sequence: the hash ids [position][table][column], each from 0 in its table.
    cache carries the compressed ids across calls, as the module's buffer does; a position no call wrote reads as
    DEAD."""
    pad = token_map[config.pad]
    for i, (t, keep) in enumerate(zip(input_ids, token_mask, strict=True)):
        cache[start_pos + i] = token_map[t] if keep else DEAD
    out: list[list[list[int]]] = []
    for i in range(len(input_ids)):
        position = start_pos + i
        tokens: list[int] = []
        blocked = False
        for shift in range(config.max_ngram):
            source = cache.get(max(position - shift, 0), DEAD)
            blocked = blocked or position < shift or source == DEAD
            tokens.append(pad if blocked else source)
        tables: list[list[int]] = []
        for mult, primes, offsets in zip(lay.multipliers, lay.primes, lay.offsets, strict=True):
            products = [t * m for t, m in zip(tokens, mult, strict=True)]
            assert all(p < 2**63 for p in products)  # the reference's int64 products do not overflow
            rolling = products[0]
            hashes: list[int] = []
            for n in range(1, config.max_ngram):
                rolling ^= products[n]
                hashes.extend(rolling % q for q in primes[(n - 1) * config.heads : n * config.heads])
            tables.append([h + o for h, o in zip(hashes, offsets, strict=True)])
        out.append(tables)
    return out


def window_rows(window: dict[str, Any], token_map: dict[int, int], config: Config, lay: Layout) -> list[int]:
    """A window's rows in the file, table 0's columns then table 1's: the hash of its last token, run over the
    sequence from the window's first position."""
    pos = int(window["pos"])
    tokens = cast("list[int]", window["tokens"])  # positions pos, pos - 1, pos - 2, pos - 3; -1 before position 0
    dead = cast("list[bool]", window["dead"])
    start = max(0, pos - (config.max_ngram - 1))
    ids = [tokens[pos - p] for p in range(start, pos + 1)]
    keep = [not dead[pos - p] for p in range(start, pos + 1)]
    last = ngram_hashes(ids, start, keep, {}, token_map=token_map, config=config, lay=lay)[-1]
    bases = lay.bases()
    return [r + bases[e] for e, table in enumerate(last) for r in table]


def load() -> dict[str, Any]:
    with ROWS.open() as f:
        return cast("dict[str, Any]", json.load(f))


def config_of(data: dict[str, Any]) -> Config:
    return Config(
        tuple(cast("list[int]", data["layers"])),
        int(data["ngram_size"]),
        int(data["heads_per_ngram"]),
        int(data["vocab_size"]),
        int(data["compressed_vocab_size"]),
        int(data["pad_token_id"]),
    )


def token_map_of(data: dict[str, Any]) -> dict[int, int]:
    return {int(t): int(c) for t, c in cast("list[list[int]]", data["token_map"])}


def test_the_files_numbers_are_the_references():
    """The file's multipliers, moduli and offsets are those engram.py derives from its settings."""
    data = load()
    lay = layout(config_of(data))
    assert [m for t in lay.multipliers for m in t] == data["layer_multipliers"]
    assert [p for t in lay.primes for p in t] == data["head_vocab_sizes"]
    bases = lay.bases()
    assert [o + bases[e] for e, t in enumerate(lay.offsets) for o in t] == data["head_offsets"]
    assert sum(sum(p) for p in lay.primes) == TABLE_ROWS


def test_the_rows_are_the_ports():
    data = load()
    config, token_map = config_of(data), token_map_of(data)
    lay = layout(config)
    windows = cast("list[dict[str, Any]]", data["windows"])
    assert len(windows) == WINDOWS
    for w in windows:
        assert window_rows(w, token_map, config, lay) == w["rows"], w


def test_the_windows_cover_the_start_the_pad_and_dead_tokens():
    data = load()
    windows = cast("list[dict[str, Any]]", data["windows"])
    for p in range(3):
        assert sum(w["pos"] == p for w in windows) >= 20
    for shift in range(4):
        assert sum(w["dead"][shift] for w in windows) >= 20
        assert sum(w["tokens"][shift] == data["pad_token_id"] and not w["dead"][shift] for w in windows) >= 20
    assert sum(not any(w["dead"]) and w["pos"] >= 3 for w in windows) >= 400
    assert token_map_of(data)[data["pad_token_id"]] == data["pad_token_id"]  # the load's check, token_map[2] == 2


def windows(seed: int = 41) -> list[dict[str, Any]]:
    """WINDOWS windows: 60 at each of positions 0-2, the rest anywhere in 2^20; each earlier token the pad token or an
    image's token one time in 12, otherwise any token."""
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for i in range(WINDOWS):
        pos = i % 3 if i < 180 else rng.randrange(3, 1 << 20)
        tokens: list[int] = []
        dead: list[bool] = []
        for shift in range(4):
            r = rng.randrange(12)
            if pos - shift < 0:
                tokens.append(-1)
                dead.append(False)
            elif r == 0:
                tokens.append(IMAGE_TOKEN)
                dead.append(True)
            else:
                tokens.append(2 if r == 1 else rng.randrange(129_280))
                dead.append(False)
        out.append({"pos": pos, "tokens": tokens, "dead": dead})
    return out


def write_rows(first_shard: Path, out: Path = ROWS) -> None:
    """Writes the JSON from the model's first shard, which holds the metadata."""
    import gguf  # noqa: PLC0415

    reader = gguf.GGUFReader(first_shard)

    def meta(key: str) -> Any:
        field = reader.get_field(f"deepseek41.{key}")
        if field is None:
            raise SystemExit(f"{first_shard} has no deepseek41.{key}")
        return field.contents()

    full_map = cast("list[int]", meta("engram.token_map"))
    data: dict[str, Any] = {
        "layers": meta("ple.layers"),
        "ngram_size": meta("ple.ngram_size"),
        "heads_per_ngram": meta("ple.heads_per_ngram"),
        "vocab_size": meta("engram.vocab_size"),
        "compressed_vocab_size": meta("engram.compressed_vocab_size"),
        "pad_token_id": meta("engram.pad_token_id"),
        "layer_multipliers": meta("ple.layer_multipliers"),
        "head_offsets": meta("ple.head_offsets"),
        "head_vocab_sizes": meta("ple.head_vocab_sizes"),
    }
    config = config_of(data)
    lay = layout(config)
    ws = windows()
    used = sorted({t for w in ws for t in w["tokens"] if t >= 0} | {config.pad})
    token_map = {t: full_map[t] for t in used}
    data["token_map"] = [[t, token_map[t]] for t in used]
    for w in ws:
        w["rows"] = window_rows(w, token_map, config, lay)
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}," for k, v in data.items()]
    rows = [f"    {json.dumps(w)}" for w in ws]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("{\n" + "\n".join(lines) + '\n  "windows": [\n' + ",\n".join(rows) + "\n  ]\n}\n")


if __name__ == "__main__":
    write_rows(Path(sys.argv[1]))
