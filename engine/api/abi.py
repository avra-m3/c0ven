"""A hand-written mirror of `include/engine_api.h`, the C ABI, version 7 (ADR-0024, item 6).

This file and the header must agree byte for byte. `layout()` lists every struct's size and every field's offset in
the form the core's `engine_abi_layout` returns them, and tests/host/test_abi.py compares the two.

Rules from DESIGN.md, section 3: only plain structs and buffers cross; every call returns a status; C++ never calls
back. Symbols carry the prefix `engine_` (ADR-0024, item 2).

The header is the core as it was optimised, not DESIGN.md's interface: no tensor table, page table or stop list
crosses, a plan is one step, and the core samples and stops requests itself. `native.py` adapts it to the interface
in `interface.py`; ADR-0027 lists each difference as a decision still to be made. Images (ADR-0031) add the
encoder's file to the config, its limit to the info, and `engine_admit_images`. Sleep (ADR-0032) adds
`engine_sleep` and `engine_wake`. Version 5 pages the KV cache (ADR-0033): Python keeps the page table and sends its
changes in each plan, a slot without a page for its next position stalls, and a checkpoint holds only the fixed state.
Version 6 adds `engine_image.frames` for video (ADR-0034). Version 7 adds min-p and the penalties to
`engine_settings`.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from enum import IntEnum

ABI_VERSION = 8

MAX_STOP = 8
MAX_STEP_SLOTS = 8
MAX_PROMPT_PIECES = 4
NO_PAGE = 0xFFFFFFFF
"""In a page change: unmap the logical page and every page after it."""


class Status(IntEnum):
    """`engine_status`."""

    OK = 0
    ERROR = 1
    BAD_ARGUMENT = 2
    BUSY = 3
    TIMEOUT = 4


class ConfigFlag(IntEnum):
    EXPERTS_SET = 1
    PROMPT_SET = 2
    SWAPS = 4
    SHARE_KV = 8


class CEventKind(IntEnum):
    """`engine_event_kind`."""

    TOKEN = 1
    """`token` sampled for `slot`; `value` its position."""
    PROMPT = 2
    """`slot` read prompt tokens; `value` how many so far."""
    DONE = 3
    """`slot` finished; `value` a CDone."""
    STEP = 4
    """A plan finished; `value` its sequence number, from 1."""
    ERROR = 5
    """`value` a Status; `engine_last_error` says more."""
    STALLED = 6
    """`slot` needs a page for position `value`; sent once until a plan maps more."""


class CDone(IntEnum):
    STOP = 1
    LENGTH = 2
    CONTEXT = 3


class CTimingKind(IntEnum):
    PROMPT = 1
    DECODE = 2
    ROUND = 3


class CConfig(ctypes.Structure):
    _fields_ = (
        ("model_path", ctypes.c_char_p),
        ("slots", ctypes.c_uint32),
        ("max_ctx", ctypes.c_uint32),
        ("max_cards", ctypes.c_uint32),
        ("experts_per_layer", ctypes.c_int32),
        ("cpu_threads", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserve_bytes", ctypes.c_uint64),
        ("prompt_chunk", ctypes.c_uint32),
        ("checkpoints", ctypes.c_uint32),
        ("routing_profile", ctypes.c_char_p),
        ("mtp_dir", ctypes.c_char_p),
        ("draft_max", ctypes.c_uint32),
        ("draft_min_p", ctypes.c_float),
        ("vision_path", ctypes.c_char_p),
        ("kv_pages", ctypes.c_uint32),
    )


class CInfo(ctypes.Structure):
    _fields_ = (
        ("api_version", ctypes.c_uint32),
        ("cards", ctypes.c_uint32),
        ("slots", ctypes.c_uint32),
        ("max_ctx", ctypes.c_uint32),
        ("experts_per_layer", ctypes.c_uint32),
        ("vocab", ctypes.c_uint32),
        ("device_bytes", ctypes.c_uint64 * 2),
        ("arena_bytes", ctypes.c_uint64),
        ("load_seconds", ctypes.c_double),
        ("prompt_chunk", ctypes.c_uint32),
        ("checkpoints", ctypes.c_uint32),
        ("max_step_rows", ctypes.c_uint32),
        ("drafting", ctypes.c_uint32),
        ("image_tokens_max", ctypes.c_uint32),
        ("kv_pages", ctypes.c_uint32),
        ("pages_per_slot", ctypes.c_uint32),
    )


class CSettings(ctypes.Structure):
    _fields_ = (
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("top_p", ctypes.c_float),
        ("max_tokens", ctypes.c_uint32),
        ("seed", ctypes.c_uint64),
        ("n_stop", ctypes.c_uint32),
        ("stop", ctypes.c_int32 * MAX_STOP),
        ("min_p", ctypes.c_float),
        ("penalty_last_n", ctypes.c_uint32),
        ("penalty_repeat", ctypes.c_float),
        ("penalty_freq", ctypes.c_float),
        ("penalty_present", ctypes.c_float),
    )


class CPromptPiece(ctypes.Structure):
    _fields_ = (
        ("slot", ctypes.c_uint32),
        ("tokens", ctypes.c_uint32),
    )


class CPageChange(ctypes.Structure):
    """`engine_page_change`: one entry of a slot's page table, applied before the plan's pieces."""

    _fields_ = (
        ("slot", ctypes.c_uint32),
        ("logical", ctypes.c_uint32),
        ("page", ctypes.c_uint32),
        ("source", ctypes.c_uint32),
        ("cells", ctypes.c_uint32),
    )


class CStepPlan(ctypes.Structure):
    _fields_ = (
        ("nslots", ctypes.c_uint32),
        ("slots", ctypes.c_uint32 * MAX_STEP_SLOTS),
        ("npieces", ctypes.c_uint32),
        ("pieces", CPromptPiece * MAX_PROMPT_PIECES),
        ("npages", ctypes.c_uint32),
        ("pages", ctypes.POINTER(CPageChange)),
        ("kv_free", ctypes.c_uint32),
    )


class CEvent(ctypes.Structure):
    _fields_ = (
        ("kind", ctypes.c_uint32),
        ("slot", ctypes.c_uint32),
        ("token", ctypes.c_int32),
        ("value", ctypes.c_uint32),
    )


class CTiming(ctypes.Structure):
    _fields_ = (
        ("kind", ctypes.c_uint32),
        ("plan", ctypes.c_uint32),
        ("rows", ctypes.c_uint32),
        ("misses", ctypes.c_uint32),
        ("start_ns", ctypes.c_uint64),
        ("end_ns", ctypes.c_uint64),
    )


class CImage(ctypes.Structure):
    """`engine_image`: an image's or frame pair's block in an admitted prompt and its pixels in the encoder's order."""

    _fields_ = (
        ("first", ctypes.c_uint32),
        ("grid_h", ctypes.c_uint32),
        ("grid_w", ctypes.c_uint32),
        ("frames", ctypes.c_uint32),
        ("pixels", ctypes.POINTER(ctypes.c_uint8)),
    )


class CAbiRow(ctypes.Structure):
    """One row of the table `engine_abi_layout` fills: "struct.field" with its offset and size, or "struct" with
    offset 0 and its size."""

    _fields_ = (
        ("name", ctypes.c_char_p),
        ("offset", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
    )


STRUCTS: dict[str, type[ctypes.Structure]] = {
    "engine_config": CConfig,
    "engine_info": CInfo,
    "engine_settings": CSettings,
    "engine_page_change": CPageChange,
    "engine_prompt_piece": CPromptPiece,
    "engine_step_plan": CStepPlan,
    "engine_event": CEvent,
    "engine_timing": CTiming,
    "engine_image": CImage,
    "engine_abi_row": CAbiRow,
}


@dataclass(frozen=True, slots=True, order=True)
class LayoutRow:
    name: str
    """"struct.field", or "struct" for the struct's own size."""
    offset: int
    size: int


def layout() -> list[LayoutRow]:
    """Every struct's size and each field's offset and size, in the form `engine_abi_layout` returns."""
    rows: list[LayoutRow] = []
    for name, struct in STRUCTS.items():
        rows.append(LayoutRow(name, 0, ctypes.sizeof(struct)))
        for field_name, *_ in struct._fields_:  # pyright: ignore[reportPrivateUsage]
            descriptor = getattr(struct, field_name)
            rows.append(LayoutRow(f"{name}.{field_name}", descriptor.offset, descriptor.size))
    return sorted(rows)


# DESIGN.md's shared timing region (section 7), which tools/attach.py reads. The header has no such region: the
# core hands out timing records through `engine_read_timing` instead (ADR-0027, item 10).

TIMING_REGION_VERSION = 1
TIMING_MAGIC = int.from_bytes(b"ENGTIME1", "little")


class CTimingRegion(ctypes.Structure):
    """The head of the timing region, a memfd the attach tool maps (DESIGN.md, section 7). `n_rings` rings
    follow it, each a CTimingRing and then `ring_records` records of TIMING_DTYPE."""

    _fields_ = (
        ("magic", ctypes.c_uint64),
        ("abi_version", ctypes.c_uint32),
        ("n_rings", ctypes.c_uint32),
        ("ring_records", ctypes.c_uint32),
        # Written by the attach tool, read by the engine once per step: 0 always on, 1 traced lists and misses.
        ("detail", ctypes.c_uint32),
        ("clock_ns_per_tick", ctypes.c_double),
    )


class CTimingRing(ctypes.Structure):
    """One thread's ring. The thread is its only writer: it writes a record, then stores `head` with release
    ordering. A full ring overwrites its oldest records; a reader that finds `head - tail` above the ring's size has
    lost the difference."""

    _fields_ = (
        ("head", ctypes.c_uint64),
        ("dropped", ctypes.c_uint64),
        ("thread", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("reserved", ctypes.c_uint64),
    )
