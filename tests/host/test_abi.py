"""The C ABI's layouts, checked from the Python side, and against the core when it is built (ADR-0024, item 6)."""

from __future__ import annotations

import ctypes
import itertools

import pytest

from engine.api import abi, native


def test_every_struct_is_listed_with_its_fields():
    rows = abi.layout()
    for name, struct in abi.STRUCTS.items():
        assert abi.LayoutRow(name, 0, ctypes.sizeof(struct)) in rows
        fields = {r.name for r in rows if r.name.startswith(f"{name}.")}
        assert fields == {f"{name}.{f[0]}" for f in struct._fields_}  # pyright: ignore[reportPrivateUsage]


def test_fields_do_not_overlap():
    rows = abi.layout()
    for name in abi.STRUCTS:
        spans = sorted((r.offset, r.offset + r.size) for r in rows if r.name.startswith(f"{name}."))
        for (_, end), (start, _) in itertools.pairwise(spans):
            assert end <= start, name


def test_sizes_from_the_header():
    """Sizes worked out by hand from engine_api.h, on x86-64."""
    assert ctypes.sizeof(abi.CConfig) == 88
    assert ctypes.sizeof(abi.CInfo) == 88
    assert ctypes.sizeof(abi.CSettings) == 80
    assert ctypes.sizeof(abi.CPageChange) == 20
    assert (
        ctypes.sizeof(abi.CStepPlan) == 4 + 4 * abi.MAX_STEP_SLOTS + 4 + 8 * abi.MAX_PROMPT_PIECES + 4 + 4 + 8 + 4 + 4
    )
    assert ctypes.sizeof(abi.CEvent) == 16
    assert ctypes.sizeof(abi.CTiming) == 32
    assert ctypes.sizeof(abi.CImage) == 24
    assert ctypes.sizeof(abi.CStats) == 8 * 38
    assert ctypes.sizeof(abi.CSlotStats) == 8 * 7
    assert ctypes.sizeof(abi.CTimingRing) == 32


def test_against_the_core():
    """The core's own table, when the core is built (build/core/libengine.so, or ENGINE_LIBRARY)."""
    path = native.library_path()
    if not path.exists():
        pytest.skip(f"the core is not built: {path} is missing")
    lib = native.load_library(path)
    assert native.core_layout(lib) == abi.layout()
