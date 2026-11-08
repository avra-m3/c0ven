"""The drafter's size on each card, from its tensor shapes and the rules the core loads them by.

The MTP layer (core/models/qwen4exp/mtp.cpp, `load_mtp`) keeps its tensors as BF16 files, each listed in the
directory's manifest with its shape; DFlash 2 (dflash.cpp, `load_dflash`) is the published `model.safetensors`, whose
header lists them (MODEL_FILES.md). The core splits the
attention and MLP projections over the cards as a main layer's, widens the norms to F32, quantises the MTP layer's
routed experts to IQ4_NL, and keeps the rest whole. The committed docs/model/mtp-manifest.json holds the MTP layer's
manifest, so the plan has its figure without the directory.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any, cast

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
DFLASH_FILES = ("model.safetensors", "config.json")
"""DFlash 2's files in its directory, as incoai/Qwen3.8-Flash-Next-DFlash2 publishes them (core/models/qwen4exp/model.h,
kDfWeights and kDfConfig)."""
DFLASH_UNUSED = "candidate_selector."
"""The checkpoint's candidate selector, which the core does not load (ADR-0050 item 5)."""
DFLASH_RING = 2112
"""K and V cells a slot keeps per DFlash 2 layer (core/models/qwen4exp/layout.h, DF_RING)."""
Q8_BLOCK = (32, 34)
"""The 8-bit K and V and indexer keys on the cards, the MTP layer's too: 34 bytes per 32 values (layout.h, Q8_BLOCK)."""
IQ4_NL_BLOCK = (32, 18)
"""The MTP layer's experts on the cards: 18 bytes per 32 values."""


def read_shapes(directory: Path = COMMITTED) -> dict[str, tuple[int, ...]]:
    """Each tensor's shape the core loads: from DFlash 2's `model.safetensors` header, leaving out the candidate
    selector's, or from the MTP layer's `mtp-manifest.json`, a map of names."""
    weights = directory / DFLASH_FILES[0]
    if weights.is_file():
        with weights.open("rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = cast("dict[str, dict[str, Any]]", json.loads(f.read(n)))
        return {
            name: tuple(t["shape"])
            for name, t in header.items()
            if name != "__metadata__" and not name.startswith(DFLASH_UNUSED)
        }
    f = directory / "mtp-manifest.json"
    if f.is_file():
        return {n: tuple(t["shape"]) for n, t in json.loads(f.read_text()).items()}
    if (directory / "dflash-manifest.json").is_file():
        raise FileNotFoundError(
            f"{directory} holds DFlash 2 split one file a tensor, which is no longer read: put the drafter's "
            "model.safetensors and config.json there (MODEL_FILES.md)"
        )
    raise FileNotFoundError(f"no drafter in {directory}")


def draft_kv_cell(shapes: dict[str, tuple[int, ...]]) -> int:
    """The values of K and V the MTP layer keeps per cell: its k and v projections' rows. DFlash 2's ring is its own
    (dflash.cpp); for it this gives the MTP layer's, from the committed manifest, as the plan has always counted."""
    if not any(n.startswith("mtp.") for n in shapes):
        shapes = read_shapes()
    return sum(shapes[f"mtp.layers.0.self_attn.{p}_proj.weight"][0] for p in ("k", "v"))


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
