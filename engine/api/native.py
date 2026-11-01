"""The native engine: the C++ core, loaded with ctypes (ADR-0022, item 4), behind the engine interface.

Two parts. `CtypesCore` is the header's calls, one method each, raising `EngineError` on a failed status.
`NativeEngine` adapts version 9 of the header to `interface.Engine`, which the scheduler was written against
(DESIGN.md, section 3). Where the two differ, the adapter bridges them as ADR-0027 lists, item by item:

- The core runs one step per plan and decodes only the slots a plan names; the interface's core keeps decoding
  between plans. The adapter keeps up to `pipeline` plans queued, each naming every slot in decode.
- A slot joins decode once the plan holding the last piece of its prompt is queued, so the core never reads a
  prompt one token per step.
- The interface's page changes go to the core in the first plan queued for the step plan that carries them, so they
  are applied before its prompt pieces (ADR-0033). A plan with page changes and no pieces is a decode step like the
  ones `pump` queues.
- Prompt spans become prompt pieces of at most the core's `prompt_chunk`, at most four to a plan. The core does not
  take a position, so the adapter checks that each span starts where the slot's last one ended.
- A stop in a plan finishes the slot at once (FINISHED, CANCELLED) and leaves decode; its later events are dropped.
- `checkpoint` and `restore` block until the queued plans have run, and the adapter answers CHECKPOINT_DONE itself.
- A slot is admitted again only after every queued plan that named it has run.
- STEP_DONE's rows and time come from the core's timing records, and so do its chunks: the prompt records of more
  than WINDOW_ROWS rows.

ctypes releases the GIL for every call through a CDLL, so `engine_wait` never holds up the protocol threads.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from engine.api import abi
from engine.api.interface import (
    NO_SLOT,
    TIMING_DTYPE,
    WINDOW_ROWS,
    EngineError,
    EngineFacts,
    EngineStats,
    ErrorCode,
    Event,
    EventKind,
    FinishReason,
    Image,
    OpenConfig,
    PageChange,
    PageOp,
    SlotSettings,
    SlotStats,
    StepPlan,
    TimingRecords,
    TimingStage,
)
from engine.host import Card

log = logging.getLogger("engine.native")

LIBRARY_ENV = "ENGINE_LIBRARY"
DEFAULT_LIBRARY = Path("build/core/libengine.so")

_u32 = ctypes.c_uint32
_p_u32 = ctypes.POINTER(ctypes.c_uint32)
_ctx = ctypes.c_void_p

_STATUS_CODES = {
    abi.Status.ERROR: ErrorCode.INTERNAL,
    abi.Status.BAD_ARGUMENT: ErrorCode.INVALID_ARGUMENT,
    abi.Status.BUSY: ErrorCode.WRONG_STATE,
    abi.Status.TIMEOUT: ErrorCode.INTERNAL,
}

_FINISH = {
    abi.CDone.STOP: FinishReason.EOS,
    abi.CDone.LENGTH: FinishReason.LENGTH,
    abi.CDone.CONTEXT: FinishReason.LENGTH,
}


def _page_changes(changes: Sequence[PageChange]) -> list[abi.CPageChange]:
    out: list[abi.CPageChange] = []
    for ch in changes:
        match ch.op:
            case PageOp.ASSIGN:
                out.append(abi.CPageChange(ch.slot, ch.logical, ch.physical, 0, 0))
            case PageOp.COPY:
                out.append(abi.CPageChange(ch.slot, ch.logical, ch.physical, ch.source, ch.cells))
            case PageOp.UNASSIGN:
                out.append(abi.CPageChange(ch.slot, ch.logical, abi.NO_PAGE, 0, 0))
    return out


def library_path() -> Path:
    return Path(os.environ.get(LIBRARY_ENV, str(DEFAULT_LIBRARY)))


def load_library(path: Path | None = None) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path or library_path()))
    signatures: dict[str, tuple[list[type], type | None]] = {
        "engine_open": ([ctypes.POINTER(abi.CConfig), ctypes.POINTER(abi.CInfo), ctypes.POINTER(_ctx)], ctypes.c_int),
        "engine_admit": (
            [_ctx, _u32, ctypes.POINTER(abi.CSettings), ctypes.POINTER(ctypes.c_int32), _u32],
            ctypes.c_int,
        ),
        "engine_admit_images": ([_ctx, _u32, ctypes.POINTER(abi.CImage), _u32], ctypes.c_int),
        "engine_release": ([_ctx, _u32], ctypes.c_int),
        "engine_submit": ([_ctx, ctypes.POINTER(abi.CStepPlan)], ctypes.c_int),
        "engine_poll": ([_ctx, ctypes.POINTER(abi.CEvent), _u32, _p_u32], ctypes.c_int),
        "engine_wait": ([_ctx, _u32], ctypes.c_int),
        "engine_checkpoint": ([_ctx, _u32, _u32], ctypes.c_int),
        "engine_restore": ([_ctx, _u32, _u32], ctypes.c_int),
        "engine_drop": ([_ctx, _u32], ctypes.c_int),
        "engine_read_timing": ([_ctx, ctypes.POINTER(abi.CTiming), _u32, _p_u32], ctypes.c_int),
        "engine_read_stats": ([_ctx, ctypes.POINTER(abi.CStats)], ctypes.c_int),
        "engine_read_slot_stats": ([_ctx, _u32, ctypes.POINTER(abi.CSlotStats)], ctypes.c_int),
        "engine_sleep": ([_ctx], ctypes.c_int),
        "engine_wake": ([_ctx, ctypes.POINTER(abi.CInfo)], ctypes.c_int),
        "engine_last_error": ([_ctx], ctypes.c_char_p),
        "engine_close": ([_ctx], None),
        "engine_abi_layout": ([ctypes.POINTER(abi.CAbiRow), _u32], _u32),
        "engine_cards": ([ctypes.POINTER(abi.CCard), _u32, _p_u32], ctypes.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


def core_layout(lib: ctypes.CDLL) -> list[abi.LayoutRow]:
    """The layout table the core was compiled with, to compare with `abi.layout()`."""
    n = int(lib.engine_abi_layout(None, 0))
    rows = (abi.CAbiRow * n)()
    lib.engine_abi_layout(rows, n)
    return sorted(abi.LayoutRow((r.name or b"").decode(), int(r.offset), int(r.size)) for r in rows)


def cards(path: Path | None = None) -> list[Card]:
    """The cards the core sees, before any engine opens. Raises EngineError when the core finds none."""
    lib = load_library(path)
    if core_layout(lib) != abi.layout():
        raise EngineError(ErrorCode.ABI_MISMATCH, "the core's struct layouts differ from engine/api/abi.py")
    n = _u32()
    if lib.engine_cards(None, 0, ctypes.byref(n)) == abi.Status.OK and n.value:
        found = (abi.CCard * n.value)()
        if lib.engine_cards(found, n.value, ctypes.byref(n)) == abi.Status.OK:
            return [
                Card(
                    c.name.decode(errors="replace"),
                    c.pci.decode(errors="replace"),
                    int(c.memory),
                    int(c.device_id),
                    int(c.ip_version),
                    int(c.xe_cores),
                    int(c.pcie_gen),
                    int(c.pcie_width),
                    int(c.pcie_bytes_per_s),
                )
                for c in found[: n.value]
            ]
    raise EngineError(ErrorCode.INTERNAL, f"no card: {(lib.engine_last_error(None) or b'').decode(errors='replace')}")


class Core(Protocol):
    """The header's calls, as `NativeEngine` uses them. `CtypesCore` is the core; the host tests use a fake."""

    def open(self, config: abi.CConfig) -> abi.CInfo: ...
    def admit(self, slot: int, settings: abi.CSettings, prompt: Sequence[int]) -> None: ...
    def admit_images(self, slot: int, images: Sequence[Image]) -> None: ...
    def release(self, slot: int) -> None: ...
    def submit(self, plan: abi.CStepPlan) -> bool:
        """False if the core's plan queue is full."""
        ...

    def poll(self, cap: int) -> list[abi.CEvent]: ...
    def wait(self, timeout_ms: int) -> bool: ...
    def checkpoint(self, slot: int, checkpoint_id: int) -> None: ...
    def restore(self, slot: int, checkpoint_id: int) -> None: ...
    def drop(self, checkpoint_id: int) -> None: ...
    def read_timing(self, cap: int) -> list[abi.CTiming]: ...
    def read_stats(self) -> abi.CStats: ...
    def slot_stats(self, slot: int) -> abi.CSlotStats:
        """Raises EngineError for a slot not admitted."""
        ...

    def sleep(self) -> None: ...
    def wake(self) -> abi.CInfo: ...
    def last_error(self) -> str: ...
    def close(self) -> None: ...


class CtypesCore:
    def __init__(self, path: Path | None = None) -> None:
        self._lib = load_library(path)
        if core_layout(self._lib) != abi.layout():
            raise EngineError(ErrorCode.ABI_MISMATCH, "the core's struct layouts differ from engine/api/abi.py")
        self._ctx = _ctx()
        self._events = (abi.CEvent * 1024)()
        self._timing = (abi.CTiming * 4096)()

    def _check(self, status: int, call: str) -> None:
        if status != abi.Status.OK:
            message = self._lib.engine_last_error(self._ctx)
            code = _STATUS_CODES.get(abi.Status(status), ErrorCode.INTERNAL)
            raise EngineError(code, f"{call}: {(message or b'').decode(errors='replace')}")

    def last_error(self) -> str:
        """What the core's last failure said: for an ERROR event, the exception that ended the step."""
        return (self._lib.engine_last_error(self._ctx) or b"").decode(errors="replace")

    def open(self, config: abi.CConfig) -> abi.CInfo:
        info = abi.CInfo()
        self._check(self._lib.engine_open(ctypes.byref(config), ctypes.byref(info), ctypes.byref(self._ctx)), "open")
        return info

    def admit(self, slot: int, settings: abi.CSettings, prompt: Sequence[int]) -> None:
        ids = (ctypes.c_int32 * len(prompt))(*prompt)
        self._check(self._lib.engine_admit(self._ctx, slot, ctypes.byref(settings), ids, len(prompt)), "admit")

    def admit_images(self, slot: int, images: Sequence[Image]) -> None:
        arr = (abi.CImage * len(images))()
        keep: list[ctypes.Array[ctypes.c_uint8]] = []
        for c, im in zip(arr, images, strict=True):
            buf = (ctypes.c_uint8 * len(im.pixels)).from_buffer_copy(im.pixels)
            keep.append(buf)
            c.first, c.grid_h, c.grid_w, c.frames = im.first, im.grid_h, im.grid_w, im.frames
            c.pixels = ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))
        self._check(self._lib.engine_admit_images(self._ctx, slot, arr, len(images)), "admit_images")

    def release(self, slot: int) -> None:
        self._check(self._lib.engine_release(self._ctx, slot), "release")

    def submit(self, plan: abi.CStepPlan) -> bool:
        status = self._lib.engine_submit(self._ctx, ctypes.byref(plan))
        if status == abi.Status.BUSY:
            return False
        self._check(status, "submit")
        return True

    def poll(self, cap: int) -> list[abi.CEvent]:
        n = _u32()
        cap = min(cap, len(self._events))
        self._check(self._lib.engine_poll(self._ctx, self._events, cap, ctypes.byref(n)), "poll")
        return [abi.CEvent.from_buffer_copy(self._events[i]) for i in range(n.value)]

    def wait(self, timeout_ms: int) -> bool:
        status = self._lib.engine_wait(self._ctx, timeout_ms)
        if status == abi.Status.TIMEOUT:
            return False
        self._check(status, "wait")
        return True

    def checkpoint(self, slot: int, checkpoint_id: int) -> None:
        self._check(self._lib.engine_checkpoint(self._ctx, slot, checkpoint_id), "checkpoint")

    def restore(self, slot: int, checkpoint_id: int) -> None:
        self._check(self._lib.engine_restore(self._ctx, slot, checkpoint_id), "restore")

    def drop(self, checkpoint_id: int) -> None:
        self._check(self._lib.engine_drop(self._ctx, checkpoint_id), "drop")

    def read_timing(self, cap: int) -> list[abi.CTiming]:
        n = _u32()
        cap = min(cap, len(self._timing))
        self._check(self._lib.engine_read_timing(self._ctx, self._timing, cap, ctypes.byref(n)), "read_timing")
        return [abi.CTiming.from_buffer_copy(self._timing[i]) for i in range(n.value)]

    def read_stats(self) -> abi.CStats:
        out = abi.CStats()
        self._check(self._lib.engine_read_stats(self._ctx, ctypes.byref(out)), "read_stats")
        return out

    def slot_stats(self, slot: int) -> abi.CSlotStats:
        out = abi.CSlotStats()
        self._check(self._lib.engine_read_slot_stats(self._ctx, slot, ctypes.byref(out)), "slot_stats")
        return out

    def sleep(self) -> None:
        self._check(self._lib.engine_sleep(self._ctx), "sleep")

    def wake(self) -> abi.CInfo:
        info = abi.CInfo()
        self._check(self._lib.engine_wake(self._ctx, ctypes.byref(info)), "wake")
        return info

    def close(self) -> None:
        self._lib.engine_close(self._ctx)
        self._ctx = _ctx()


@dataclass(slots=True)
class _Slot:
    prompt_len: int
    fed: int = 0
    """Prompt positions queued so far as pieces, or restored."""
    spans: deque[tuple[int, int]] = field(default_factory=deque)
    """Each queued span's end position and row count, until the core reports it read."""
    decoding: bool = False
    finished: bool = False


@dataclass(slots=True)
class _Queued:
    slots: frozenset[int]
    prompt_rows: int
    decode_rows: int


TIMING_KEPT = 1 << 16


class NativeEngine:
    def __init__(self, path: Path | None = None, *, core: Core | None = None, pipeline: int = 2) -> None:
        self._core: Core = core if core is not None else CtypesCore(path)
        self._pipeline = pipeline
        self._info: abi.CInfo | None = None
        self._asleep = False
        self._model_path = b""
        self._draft_dir = b""
        self._vision_path = b""
        self._stop_ids: tuple[int, ...] = ()
        self._slots: dict[int, _Slot] = {}
        self._checkpoint_len: dict[int, int] = {}
        self._queued: dict[int, _Queued] = {}
        self._submitted = 0
        """Plans queued in the core, numbered as the core numbers them: from 1, in order."""
        self._done = 0
        self._serial = -1
        self._kv_free = 0
        """The last plan's kv_free, for the plans the adapter queues itself."""
        self._events: deque[Event] = deque()
        self._plan_ns: dict[int, int] = {}
        self._plan_chunks: dict[int, int] = {}
        self._timing: deque[tuple[int, int, int, int]] = deque(maxlen=TIMING_KEPT)

    @property
    def info(self) -> abi.CInfo:
        if self._info is None:
            raise EngineError(ErrorCode.WRONG_STATE, "not open")
        if self._asleep:
            raise EngineError(ErrorCode.WRONG_STATE, "asleep")
        return self._info

    def open(self, config: OpenConfig) -> EngineFacts:
        if self._info is not None:
            raise EngineError(ErrorCode.WRONG_STATE, "already open")
        if not config.shard_paths:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, "no model file")
        if config.slots > abi.MAX_STEP_SLOTS:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"{config.slots} slots > {abi.MAX_STEP_SLOTS}")
        self._model_path = config.shard_paths[0].encode()
        c = abi.CConfig()
        c.model_path = self._model_path
        c.slots = config.slots
        c.max_ctx = config.max_context
        c.cpu_threads = config.cpu_workers
        c.checkpoints = config.checkpoints
        c.kv_pages = config.pages
        c.kv_resident_cells = config.kv_resident_cells
        # Adaptive swaps but with ADR-0016's switches; drafting when the draft layer is given (ADR-0029).
        if not config.deterministic:
            c.flags |= abi.ConfigFlag.SWAPS | abi.ConfigFlag.SHARE_KV
        self._draft_dir = config.draft_dir.encode()
        if self._draft_dir:
            c.mtp_dir = self._draft_dir
        self._vision_path = config.vision_path.encode()
        if self._vision_path:
            c.vision_path = self._vision_path
        info = self._core.open(c)
        if info.api_version != abi.ABI_VERSION:
            self._core.close()
            raise EngineError(ErrorCode.ABI_MISMATCH, f"core has ABI {info.api_version}, Python {abi.ABI_VERSION}")
        if info.prompt_chunk == 0:
            self._core.close()
            raise EngineError(ErrorCode.INVALID_ARGUMENT, "the core's prompt path is off")
        self._info = info
        self._stop_ids = tuple(config.eos_ids[: abi.MAX_STOP])
        log.info(
            "core open: %d card(s), %d experts per layer, prompt chunk %d, drafting %d, images up to %d tokens, "
            "swaps %d, %.1f s",
            info.cards,
            info.experts_per_layer,
            info.prompt_chunk,
            info.drafting,
            info.image_tokens_max,
            not config.deterministic,
            info.load_seconds,
        )
        return self._facts(info)

    def _facts(self, info: abi.CInfo) -> EngineFacts:
        return EngineFacts(
            cards=info.cards,
            slots=info.slots,
            max_context=info.max_ctx,
            pages=int(info.kv_pages),
            checkpoints=info.checkpoints,
            experts_cached=info.experts_per_layer,
            vram_used=tuple(int(info.device_bytes[i]) for i in range(info.cards)),
            vram_free=(),
            host_pinned=int(info.arena_bytes),
            timing_fd=-1,
            version=f"core-abi{info.api_version}",
            image_tokens_max=int(info.image_tokens_max),
            host_read_bytes_per_s=float(info.host_read_bytes_per_s),
        )

    def sleep(self) -> None:
        _ = self.info
        if self._slots:
            raise EngineError(ErrorCode.WRONG_STATE, f"slots {sorted(self._slots)} are admitted")
        while self._queued:
            self._core.wait(1000)
            self._collect(pump=False)
        self._core.sleep()
        self._asleep = True
        self._checkpoint_len.clear()
        self._events.clear()

    def wake(self) -> EngineFacts:
        if self._info is None or not self._asleep:
            raise EngineError(ErrorCode.WRONG_STATE, "not asleep")
        info = self._core.wake()
        self._info = info
        self._asleep = False
        self._serial = -1  # a new scheduler numbers its plans from the start; the core's own count goes on
        log.info("core awake: %d experts per layer, %.1f s", info.experts_per_layer, info.load_seconds)
        return self._facts(info)

    def admit(self, slot: int, settings: SlotSettings, prompt: tuple[int, ...], images: tuple[Image, ...] = ()) -> None:
        _ = self.info
        if slot in self._slots:
            raise EngineError(ErrorCode.WRONG_STATE, f"slot {slot} is in use")
        self._settle(slot)
        c = abi.CSettings()
        c.temperature = settings.temperature
        c.top_k = settings.top_k
        c.top_p = settings.top_p
        c.max_tokens = settings.max_new
        c.seed = settings.seed
        c.min_p = settings.min_p
        c.penalty_last_n = settings.penalty_last_n
        c.penalty_repeat = settings.penalty_repeat
        c.penalty_freq = settings.penalty_freq
        c.penalty_present = settings.penalty_present
        c.n_stop = len(self._stop_ids)
        for i, token in enumerate(self._stop_ids):
            c.stop[i] = token
        self._core.admit(slot, c, prompt)
        if images:
            try:
                self._core.admit_images(slot, images)
            except EngineError:
                self._core.release(slot)
                raise
        self._slots[slot] = _Slot(len(prompt))

    def release(self, slot: int) -> None:
        if self._slots.pop(slot, None) is None:
            raise EngineError(ErrorCode.WRONG_STATE, f"slot {slot} is not admitted")
        self._core.release(slot)

    def submit(self, plan: StepPlan) -> None:
        _ = self.info
        plan.check()
        if plan.serial <= self._serial:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"plan serial {plan.serial} after {self._serial}")
        self._serial = plan.serial
        self._kv_free = plan.kv_free
        for slot in plan.stop:
            s = self._slot(slot)
            if not s.finished:
                self._finish(slot, s, FinishReason.CANCELLED)
        pieces: list[tuple[int, int]] = []
        joining: list[_Slot] = []
        chunk = self.info.prompt_chunk
        for span in plan.prompt:
            s = self._slot(span.slot)
            if s.decoding or s.finished:
                raise EngineError(ErrorCode.INVALID_ARGUMENT, f"prompt span for slot {span.slot} past its prompt")
            if span.first != s.fed:
                raise EngineError(ErrorCode.INVALID_ARGUMENT, f"span starts at {span.first}, slot is at {s.fed}")
            end = span.first + span.count
            if span.count <= 0 or end > s.prompt_len or span.last != (end == s.prompt_len):
                raise EngineError(ErrorCode.INVALID_ARGUMENT, f"span {span.first}+{span.count} of {s.prompt_len}")
            for first in range(span.first, end, chunk):
                pieces.append((span.slot, min(chunk, end - first)))
            s.fed = end
            s.spans.append((end, span.count))
            if span.last:
                joining.append(s)
        changes = _page_changes(plan.pages)
        for i in range(0, len(pieces), abi.MAX_PROMPT_PIECES):
            self._queue(pieces[i : i + abi.MAX_PROMPT_PIECES], self._decode_slots(), changes)
            changes = []
        if changes:
            self._queue([], self._decode_slots(), changes)
        for s in joining:
            s.decoding = True
        self._pump()

    def checkpoint(self, slot: int, checkpoint_id: int) -> None:
        s = self._slot(slot)
        self._core.checkpoint(slot, checkpoint_id)
        self._checkpoint_len[checkpoint_id] = s.fed
        self._events.append(Event(EventKind.CHECKPOINT_DONE, slot, self._done, checkpoint_id))

    def restore(self, slot: int, checkpoint_id: int) -> None:
        s = self._slot(slot)
        self._core.restore(slot, checkpoint_id)
        s.fed = self._checkpoint_len[checkpoint_id]

    def drop(self, checkpoint_id: int) -> None:
        self._core.drop(checkpoint_id)
        self._checkpoint_len.pop(checkpoint_id, None)

    def poll(self, max_events: int = 256) -> list[Event]:
        self._collect()
        out: list[Event] = []
        while self._events and len(out) < max_events:
            out.append(self._events.popleft())
        return out

    def wait(self, timeout_s: float, max_events: int = 256) -> list[Event]:
        if not self._events:
            self._core.wait(max(0, math.ceil(timeout_s * 1e3)))
        return self.poll(max_events)

    def read_timing(self, max_records: int = 4096) -> TimingRecords:
        self._collect_timing()
        n = min(max_records, len(self._timing))
        out = np.zeros(n, dtype=TIMING_DTYPE)
        for i in range(n):
            start, duration, plan, rows = self._timing.popleft()
            out[i] = (start, duration, plan, TimingStage.STEP, 0, rows, 0)
        return out

    def stats(self) -> EngineStats:
        if self._info is None:
            return EngineStats()
        c = self._core.read_stats()
        return EngineStats(**{name: int(getattr(c, name)) for name in abi.STATS_FIELDS})

    def slot_stats(self, slot: int) -> SlotStats:
        if self._info is None or self._asleep or slot not in self._slots:
            return SlotStats()
        try:
            c = self._core.slot_stats(slot)
        except EngineError:
            return SlotStats()
        return SlotStats(**{name: int(getattr(c, name)) for name in abi.SLOT_STATS_FIELDS})

    def close(self) -> None:
        if self._info is not None:
            self._core.close()
        self._info = None
        self._asleep = False
        self._slots.clear()
        self._queued.clear()

    # The adapter.

    def _slot(self, slot: int) -> _Slot:
        s = self._slots.get(slot)
        if s is None:
            raise EngineError(ErrorCode.WRONG_STATE, f"slot {slot} is not admitted")
        return s

    def _decode_slots(self) -> list[int]:
        return sorted(slot for slot, s in self._slots.items() if s.decoding and not s.finished)

    def _finish(self, slot: int, s: _Slot, reason: FinishReason) -> None:
        s.finished = True
        s.decoding = False
        self._events.append(Event(EventKind.FINISHED, slot, self._done, reason))

    def _queue(
        self, pieces: Sequence[tuple[int, int]], decode: Sequence[int], pages: Sequence[abi.CPageChange] = ()
    ) -> None:
        c = abi.CStepPlan()
        array = (abi.CPageChange * len(pages))(*pages)  # submit copies it
        c.npages = len(pages)
        c.pages = ctypes.cast(array, ctypes.POINTER(abi.CPageChange))
        c.npieces = len(pieces)
        for i, (slot, tokens) in enumerate(pieces):
            c.pieces[i].slot = slot
            c.pieces[i].tokens = tokens
        c.nslots = len(decode)
        for i, slot in enumerate(decode):
            c.slots[i] = slot
        c.kv_free = self._kv_free
        while not self._core.submit(c):
            self._core.wait(100)
            self._collect(pump=False)
        self._submitted += 1
        self._queued[self._submitted] = _Queued(
            frozenset(decode) | {slot for slot, _ in pieces} | {p.slot for p in pages},
            sum(t for _, t in pieces),
            len(decode),
        )

    def _pump(self) -> None:
        """Keep the core decoding: up to `pipeline` plans queued while any slot is in decode."""
        while self._submitted - self._done < self._pipeline and (decode := self._decode_slots()):
            self._queue([], decode)

    def _settle(self, slot: int) -> None:
        """Wait until no queued plan names the slot, so the core is not running it when it is admitted again."""
        while any(slot in q.slots for q in self._queued.values()):
            self._core.wait(1000)
            self._collect()

    def _collect_timing(self) -> None:
        for t in self._core.read_timing(4096):
            duration = int(t.end_ns - t.start_ns)
            self._plan_ns[t.plan] = self._plan_ns.get(t.plan, 0) + duration
            rows = int(t.rows) if t.kind == abi.CTimingKind.PROMPT else 0
            if rows > WINDOW_ROWS:
                self._plan_chunks[t.plan] = self._plan_chunks.get(t.plan, 0) + 1
            self._timing.append((int(t.start_ns), duration, int(t.plan), rows))

    def _collect(self, *, pump: bool = True) -> None:
        events = self._core.poll(1024)
        if events:
            self._collect_timing()
        for c in events:
            self._translate(c)
        if pump:
            self._pump()

    def _translate(self, c: abi.CEvent) -> None:
        kind = abi.CEventKind(c.kind)
        if kind is abi.CEventKind.STEP:
            self._plan_finished(int(c.value))
            return
        if kind is abi.CEventKind.ERROR:
            log.error("core error %d: %s", c.value, self._core.last_error())
            self._plan_finished(self._done + 1, failed=True)
            self._events.append(Event(EventKind.ERROR, NO_SLOT, self._done, ErrorCode.INTERNAL))
            return
        s = self._slots.get(c.slot)
        if s is None or s.finished:
            return
        match kind:
            case abi.CEventKind.STALLED:
                self._events.append(Event(EventKind.STALLED, c.slot, self._done + 1, c.value))
            case abi.CEventKind.TOKEN:
                self._events.append(Event(EventKind.TOKEN, c.slot, self._done + 1, c.token, c.value))
            case abi.CEventKind.PROMPT:
                while s.spans and c.value >= s.spans[0][0]:
                    end, rows = s.spans.popleft()
                    self._events.append(Event(EventKind.PROMPT_DONE, c.slot, self._done + 1, end, rows))
            case abi.CEventKind.DONE:
                self._finish(c.slot, s, _FINISH.get(abi.CDone(c.value), FinishReason.LENGTH))

    def _plan_finished(self, seq: int, *, failed: bool = False) -> None:
        self._done = seq
        q = self._queued.pop(seq, None)
        ns = self._plan_ns.pop(seq, 0)
        chunks = self._plan_chunks.pop(seq, 0)
        if failed or q is None:
            return
        self._events.append(Event(EventKind.STEP_DONE, NO_SLOT, seq, q.decode_rows, q.prompt_rows, ns, chunks))
