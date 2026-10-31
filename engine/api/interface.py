"""The engine's interface, as the control plane sees it (DESIGN.md, section 3).

Three implementations stand behind it (ADR-0022, item 6): the native core through `engine_api.h`
(`native.py`), a mock in Python (`mock.py`) and a reference that replays llama.cpp's recorded output
(`reference.py`). The types here are plain Python; `abi.py` turns them into the header's structs.

The order rule: `submit`, `checkpoint`, `restore` and `drop` take effect in the order they are called. A
checkpoint called after a plan is taken after that plan's work and before the next plan's. So the scheduler can
end a prompt span at a checkpoint position, call `checkpoint`, and submit the next span without waiting.

Between plans the core keeps running: slots in decode go on decoding, step after step, with no plan submitted.
A plan carries only changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

import numpy as np
import numpy.typing as npt

PAGE_CELLS = 4
"""Cells per KV page: one indexer block (DESIGN.md, section 4). Also in the layout header the core and kernels share."""

MAX_SLOTS = 16
MAX_PROMPT_SPANS = 8
WINDOW_ROWS = 64
"""A prompt piece of at most this many rows is read as decode windows (the core's kWindowPrompt), without the prompt
path's staging of the layers' uncached experts."""
PROMPT_RUNS = 4
"""The longer pieces of each four prompt spans of a plan, in order, are read as one chunk, the experts staged once for
all their rows (the core's PROMPT_MAX_RUNS and pieces a plan)."""
MAX_PAGE_CHANGES = 2048
"""Page changes a plan carries at most: a whole 4,352-row chunk's 1,088 pages with room for the writing slots'
batches (SchedulerOptions.decode_batch). The core copies any count; each plan's changes cost it one write of the
page tables (ADR-0033)."""
MAX_EOS = 4
MAX_BATCH_WIDTHS = 8
MAX_SHARDS = 4


class EngineError(RuntimeError):
    """A call returned an error code."""

    def __init__(self, code: ErrorCode, message: str) -> None:
        super().__init__(f"{code.name}: {message}")
        self.code = code


class ErrorCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = 1
    WRONG_STATE = 2
    OUT_OF_MEMORY = 3
    DEVICE_LOST = 4
    IO = 5
    ABI_MISMATCH = 6
    INTERNAL = 7


class EventKind(IntEnum):
    TOKEN = 1
    """`a` the token, `b` its position. A slot's first token follows the last row of its prompt."""
    PROMPT_DONE = 2
    """A prompt span finished: `a` the positions now in the slot's state, `b` the span's row count."""
    FINISHED = 3
    """The slot stopped and left decode: `a` a FinishReason. The slot stays admitted until `release`."""
    STEP_DONE = 4
    """`a` decoding slots, `b` prompt rows, `c` the prompt pieces of more than WINDOW_ROWS rows (each a pass of the
    prompt path), `t_ns` the step's time on the cards."""
    CHECKPOINT_DONE = 5
    """`a` the checkpoint id."""
    STALLED = 6
    """The slot reached a position whose page was not assigned yet; it waits, and resumes when one is."""
    ERROR = 7
    """`a` an ErrorCode. `slot` is the slot, or NO_SLOT for the engine as a whole."""


NO_SLOT = 0xFFFFFFFF


class FinishReason(IntEnum):
    EOS = 1
    LENGTH = 2
    CANCELLED = 3


class SplitKind(IntEnum):
    """How a tensor is laid over the two cards (ADR-0023). The model's rules are in models/<arch>/split.py."""

    WHOLE = 0
    """Both cards hold all of it."""
    ROWS = 1
    """Each card holds half the output rows."""
    COLUMNS = 2
    """Each card holds half the input columns, giving partial sums that an exchange adds."""
    EXPERT_ROWS = 3
    """Routed experts: each cached expert's output rows are halved; misses stay whole in the host arena."""
    EXPERT_COLUMNS = 4
    """Routed experts' down projection: each card takes the input columns matching its half of gate and up."""
    FILE = 5
    """Read from the file on demand, never loaded: the n-gram table (ADR-0015)."""


class PageOp(IntEnum):
    ASSIGN = 0
    """Logical page `logical` of `slot` is physical page `physical`."""
    COPY = 1
    """Copy the first `cells` cells of physical page `source` into `physical`, then assign it as ASSIGN does.
    For the partial last page of a reused prefix."""
    UNASSIGN = 2
    """Logical page `logical` of `slot` and every page after it are no longer mapped."""


@dataclass(frozen=True, slots=True)
class Tensor:
    name: str
    ggml_type: int
    shard: int
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    split: SplitKind
    layer: int
    """-1 for tensors outside the layers."""


@dataclass(frozen=True, slots=True)
class OpenConfig:
    """What `open` is given: the files, the tensor table and the memory plan, all decided in Python."""

    shard_paths: tuple[str, ...]
    tensors: tuple[Tensor, ...]
    slots: int
    max_context: int
    pages: int
    checkpoints: int
    kv_resident_cells: int
    experts_cached: int
    arena_bytes: int
    working_bytes: int
    cpu_workers: int
    ngram_cache_rows: int
    timing_records: int
    eos_ids: tuple[int, ...]
    vocab_size: int
    batch_widths: tuple[int, ...] = (1,)
    deterministic: bool = False
    """ADR-0016's switches: no PCIe share of the misses, no adaptive swaps, no prefetching."""
    draft_dir: str = ""
    """The draft (MTP) layer's tensors, one file each (MODEL_FILES.md, "The MTP layer"); empty: no drafting."""
    vision_path: str = ""
    """The image encoder's file (the mmproj GGUF, spec/MODEL_VISION.md); empty: no images."""


@dataclass(frozen=True, slots=True)
class EngineFacts:
    """What `open` reports."""

    cards: int
    slots: int
    max_context: int
    pages: int
    checkpoints: int
    experts_cached: int
    vram_used: tuple[int, ...]
    vram_free: tuple[int, ...]
    host_pinned: int
    timing_fd: int
    """The memfd of the timing region, for the attach tool (DESIGN.md, section 7); -1 if none."""
    version: str
    image_tokens_max: int = 0
    """The most tokens an image may have; 0: no images."""
    host_read_bytes_per_s: float = 0.0
    """The CPU workers' read of the host's expert arena at the load, bytes a second; 0: not measured."""


IMAGE_PATCH_BYTES = 768
"""One frame of a patch: 16 x 16 pixels, 8-bit RGB, channel, row, column (spec/MODEL_VISION.md, section 2.2)."""


@dataclass(frozen=True, slots=True)
class Image:
    """An image, or a pair of a video's frames, in a prompt: its block of tokens starts at prompt index `first`;
    `pixels` holds its grid_h x grid_w patches in the encoder's order (merge windows row-major, each window's four
    patches row-major), each `frames` x 768 bytes, frame by frame. `frames` is 1 for a still image, whose block holds
    `<|image_pad|>`, and 2 for a frame pair, whose block holds `<|video_pad|>` (spec/MODEL_VISION.md, section 6)."""

    first: int
    grid_h: int
    grid_w: int
    pixels: bytes = field(repr=False)
    frames: int = 1

    @property
    def tokens(self) -> int:
        return (self.grid_h // 2) * (self.grid_w // 2)

    def check(self) -> None:
        if self.grid_h <= 0 or self.grid_w <= 0 or self.grid_h % 2 or self.grid_w % 2:
            raise ValueError(f"an image's grid must be even and not empty, not {self.grid_h} x {self.grid_w}")
        if self.frames not in (1, 2):
            raise ValueError(f"an image has 1 frame or 2, not {self.frames}")
        if len(self.pixels) != self.grid_h * self.grid_w * self.frames * IMAGE_PATCH_BYTES:
            raise ValueError(
                f"{len(self.pixels)} bytes of pixels for {self.grid_h} x {self.grid_w} patches of "
                f"{self.frames} frame(s)"
            )

    def digest(self) -> int:
        """A 62-bit digest of the grid, the frames and the pixels, for prefix keys."""
        h = hashlib.blake2b(digest_size=8)
        h.update(f"{self.grid_h}x{self.grid_w}x{self.frames}:".encode())
        h.update(self.pixels)
        return int.from_bytes(h.digest(), "little") >> 2


def prefix_key(prompt: tuple[int, ...], images: tuple[Image, ...]) -> tuple[int, ...]:
    """The prompt as prefix caches compare it: each image's tokens become one negative number from its digest, so two
    prompts share a prefix only where their images are the same."""
    if not images:
        return prompt
    key = list(prompt)
    for im in images:
        mark = -1 - im.digest()
        key[im.first : im.first + im.tokens] = [mark] * im.tokens
    return tuple(key)


@dataclass(frozen=True, slots=True)
class SlotSettings:
    temperature: float
    top_p: float
    top_k: int
    min_p: float
    penalty_last_n: int
    penalty_repeat: float
    penalty_freq: float
    penalty_present: float
    seed: int
    max_new: int


@dataclass(frozen=True, slots=True)
class PromptSpan:
    slot: int
    first: int
    """The first prompt position in this span."""
    count: int
    last: bool
    """The span ends the prompt: the cards sample the slot's first token and it joins decode."""


@dataclass(frozen=True, slots=True)
class PageChange:
    op: PageOp
    slot: int
    logical: int
    physical: int = 0
    source: int = 0
    cells: int = 0


@dataclass(slots=True)
class StepPlan:
    """The only thing that crosses from the control plane to the data plane (DESIGN.md, section 1)."""

    serial: int
    prompt: list[PromptSpan] = field(default_factory=lambda: [])
    pages: list[PageChange] = field(default_factory=lambda: [])
    stop: list[int] = field(default_factory=lambda: [])
    """Slots to stop now; each answers with FINISHED, CANCELLED."""
    kv_free: int = 0
    """The pages at the top of the pool nothing holds, after `pages`: the engine may lend their memory to the expert
    cache until a plan maps them again (`include/engine_api.h`)."""

    def check(self) -> None:
        if len(self.prompt) > MAX_PROMPT_SPANS:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"{len(self.prompt)} prompt spans > {MAX_PROMPT_SPANS}")
        if len(self.pages) > MAX_PAGE_CHANGES:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"{len(self.pages)} page changes > {MAX_PAGE_CHANGES}")
        if len(self.stop) > MAX_SLOTS:
            raise EngineError(ErrorCode.INVALID_ARGUMENT, f"{len(self.stop)} stops > {MAX_SLOTS}")

    @property
    def empty(self) -> bool:
        return not (self.prompt or self.pages or self.stop)


@dataclass(frozen=True, slots=True)
class Event:
    kind: EventKind
    slot: int
    step: int
    a: int = 0
    b: int = 0
    t_ns: int = 0
    c: int = 0


TIMING_DTYPE = np.dtype(
    [
        ("t", "<u8"),
        ("duration", "<u8"),
        ("step", "<u4"),
        ("stage", "<u2"),
        ("thread", "<u2"),
        ("aux", "<u4"),
        ("pad", "<u4"),
    ]
)
"""One timing record, 32 bytes (DESIGN.md, section 7). `t` and `duration` are in nanoseconds once the core has lined
the clocks up."""

type TimingRecords = npt.NDArray[np.void]


@dataclass(frozen=True, slots=True)
class EngineStats:
    """The engine's counters since open, `engine_stats` field for field (DESIGN.md, section 7): sums over everything
    the engine ran, times in nanoseconds. A reader differences two snapshots. The mock fills what it has."""

    up_ns: int = 0
    plans: int = 0
    busy_ns: int = 0
    prompt_chunks: int = 0
    prompt_rows: int = 0
    prompt_ns: int = 0
    window_pieces: int = 0
    window_rows: int = 0
    window_ns: int = 0
    steps: int = 0
    """Decode steps that verified no drafts."""
    step_rows: int = 0
    step_ns: int = 0
    step_misses: int = 0
    """Experts the CPU computed in them."""
    step_layers_missed: int = 0
    undrafted_steps: int = 0
    """Of the steps, with drafting on, those whose rows were all writing: no slot had a draft to verify."""
    rounds: int = 0
    """Draft rounds: windows verifying drafts, of one slot or several."""
    round_rows: int = 0
    round_ns: int = 0
    round_misses: int = 0
    round_layers_missed: int = 0
    layer_runs: int = 0
    """A slot's run in a round with the draft layer's drafts; `layer_drafts` verified, `layer_accepted` accepted."""
    layer_drafts: int = 0
    layer_accepted: int = 0
    layer_ns: int = 0
    """The rounds' time by the source of a lone slot's drafts; a round of several slots counts to the layer."""
    lookup_runs: int = 0
    """The same for drafts from the slot's own tokens (ADR-0041)."""
    lookup_drafts: int = 0
    lookup_accepted: int = 0
    lookup_ns: int = 0
    swap_rounds: int = 0
    swaps: int = 0
    swap_ns: int = 0
    stalls: int = 0
    checkpoints_taken: int = 0
    restores: int = 0
    ngram_rows: int = 0
    """The n-gram table's rows asked for, those its cache served, and those read from the SSD (ADR-0015)."""
    ngram_hits: int = 0
    ngram_reads: int = 0
    experts_cached: int = 0
    """Per layer at the last plan's end, the mean over the layers; `experts_floor` the fewest the load keeps room
    for, whatever the KV pool holds (ADR-0045)."""
    experts_floor: int = 0


@dataclass(frozen=True, slots=True)
class SlotStats:
    """One slot's decode counters since admit, `engine_slot_stats` field for field: its own tokens, not its prompt."""

    steps: int = 0
    """Decode steps and rounds that fed it one of its own tokens."""
    rounds: int = 0
    """Of them, draft rounds; `drafts` verified for it, `accepted` accepted."""
    drafts: int = 0
    accepted: int = 0
    lookup_rounds: int = 0
    """Of the rounds, those with drafts from its own tokens."""
    lookup_drafts: int = 0
    lookup_accepted: int = 0


class TimingStage(IntEnum):
    """What a timing record measures: its `stage` field."""

    STEP = 0
    """A whole step on the cards; `aux` its prompt rows."""
    CARD_WORK = 1
    EXCHANGE = 2
    CPU_MISS = 3
    FLAG_WAIT = 4
    NGRAM = 5
    """The host's n-gram rows for the next step (DESIGN.md, section 2)."""
    SUBMIT = 6
    POLL = 7


class Engine(Protocol):
    def open(self, config: OpenConfig) -> EngineFacts: ...
    def admit(
        self, slot: int, settings: SlotSettings, prompt: tuple[int, ...], images: tuple[Image, ...] = ()
    ) -> None: ...
    def release(self, slot: int) -> None: ...
    def submit(self, plan: StepPlan) -> None: ...
    def poll(self, max_events: int = 256) -> list[Event]: ...
    def wait(self, timeout_s: float, max_events: int = 256) -> list[Event]: ...
    def checkpoint(self, slot: int, checkpoint_id: int) -> None: ...
    def restore(self, slot: int, checkpoint_id: int) -> None: ...
    def drop(self, checkpoint_id: int) -> None: ...
    def read_timing(self, max_records: int = 4096) -> TimingRecords: ...
    def stats(self) -> EngineStats:
        """The counters since open, awake or asleep."""
        ...

    def slot_stats(self, slot: int) -> SlotStats:
        """An admitted slot's counters; zeros for one that is not."""
        ...

    def sleep(self) -> None:
        """Gives the cards back: frees every allocation on them and the checkpoints, and keeps the arena
        (ADR-0032). No slot may be admitted. Until `wake`, only `close` may follow."""
        ...

    def wake(self) -> EngineFacts:
        """Loads the cards again after `sleep`. The slots are empty and the checkpoints too."""
        ...

    def close(self) -> None: ...
