"""The drafter's size on each card, from its manifest's tensor shapes and the rules the core loads them by.

The MTP layer (core/models/qwen4exp/mtp.cpp, `load_mtp`) and DFlash 2 (dflash.cpp, `load_dflash`) keep their
tensors as BF16 files, each listed in the directory's manifest with its shape (MODEL_FILES.md). The core splits the
attention and MLP projections over the cards as a main layer's, widens the norms to F32, quantises the MTP layer's
routed experts to IQ4_NL, and keeps the rest whole. The committed docs/model/mtp-manifest.json holds the MTP layer's
manifest, so the plan has its figure without the directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

COMMITTED = Path(__file__).resolve().parents[3] / "docs" / "model"
"""Holds mtp-manifest.json, the MTP layer's manifest: the plan's drafter when it is given no directory."""

_SPLIT = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "down_proj.weight",
)
"""Split over the cards by rows (q, k, v, gate, up) or by columns (o, down): the MLP's and the MTP layer's shared
expert's."""
_UNUSED = ".indexer."
"""The MTP layer's indexer: the draft layer attends to its last cells without it."""
DFLASH_RING = 2112
"""K and V cells a slot keeps per DFlash 2 layer (core/models/qwen4exp/layout.h, DF_RING)."""
Q8_BLOCK = (32, 34)
"""The MTP layer's K and V on the cards: 34 bytes per 32 values."""
IQ4_NL_BLOCK = (32, 18)
"""The MTP layer's experts on the cards: 18 bytes per 32 values."""


def read_shapes(directory: Path = COMMITTED) -> dict[str, tuple[int, ...]]:
    """Each tensor's shape from the drafter's manifest: `mtp-manifest.json`, a map of names, or
    `dflash-manifest.json`, whose `tensors` list each carries its name."""
    for name in ("dflash-manifest.json", "mtp-manifest.json"):
        f = directory / name
        if f.is_file():
            body = json.loads(f.read_text())
            if name.startswith("dflash"):
                return {t["name"]: tuple(t["shape"]) for t in body["tensors"]}
            return {n: tuple(t["shape"]) for n, t in body.items()}
    raise FileNotFoundError(f"no drafter manifest in {directory}")


def _align(n: int) -> int:
    return -(-n // 256) * 256


def _experts(shape: tuple[int, ...], cards: int, gate_up: bool) -> int:
    """One card's share of the routed experts, as `load_mtp` lays out each expert's slot: each matrix 256-byte
    aligned. gate_up: [experts, 2 x ff, embd], else down: [experts, embd, ff]."""
    values, nbytes = IQ4_NL_BLOCK
    experts, rows, cols = shape
    if gate_up:
        return experts * 2 * _align(rows // 2 // cards * cols // values * nbytes)
    return experts * _align(rows * (cols // cards) // values * nbytes)


def kv_per_slot(shapes: dict[str, tuple[int, ...]], context: int, cards: int) -> int:
    """One slot's K and V on each card, bytes. The MTP layer holds them as Q8_0 for the whole context (`load_mtp`), and
    each DFlash 2 layer as F32 in a ring of its own (`load_dflash`)."""
    keys = [(name, shape[0]) for name, shape in shapes.items() if name.endswith("self_attn.k_proj.weight")]
    if any(name.startswith("mtp.") for name, _ in keys):
        values, nbytes = Q8_BLOCK
        return sum(2 * context * (rows // cards) // values * nbytes for _, rows in keys)
    return sum(2 * DFLASH_RING * (rows // cards) * 4 for _, rows in keys)


def draft_per_card(shapes: dict[str, tuple[int, ...]], cards: int) -> int:
    """The drafter's weights on each card."""
    total = 0
    for name, shape in shapes.items():
        if _UNUSED in name:
            continue
        values = math.prod(shape)
        if name.endswith("experts.gate_up_proj"):
            total += _experts(shape, cards, gate_up=True) + shape[0] * 4  # and the slot map
        elif name.endswith("experts.down_proj"):
            total += _experts(shape, cards, gate_up=False)
        elif "norm" in name or name.endswith("base_kernel"):  # widened from BF16 to F32 at the load
            total += values * 4
        elif name.endswith(_SPLIT):
            total += values * 2 // cards
        else:
            total += values * 2
    return total
