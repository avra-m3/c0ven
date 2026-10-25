"""The line protocol between the front end and the engine process (DESIGN.md, sections 1 and 2).

One message per line, ASCII, fields separated by single spaces. Every line about a request carries its id, so the
tokens of several requests can be told apart (ADR-0010, item 6). The lines follow Strata's
(spec/STRATA_DECODE.md, section 10), with the request id added and the fields Strata's single-sequence engine needed
for its caches left out.

Front end to engine:

    IMG <rid> <first> <grid_h> <grid_w> <frames> <path>
    GEN <rid> <max_new> [key=value ...] <id,id,...>
    STOP <rid>
    STATS
    SLEEP [<drain_ms>]
    WAKE
    QUIT

A request's images come before its GEN, one IMG line each, in prompt order; a video is one IMG line per pair of its
frames, `frames` 2. `path` names a file of the front end's (a memfd, as /proc/<pid>/fd/<n>) holding the image's
patches, which the engine reads when it reads the line.

Engine to front end:

    INFO key=value ...
    READY <max_context> <slots>
    ASLEEP <ms>
    RESUME <rid> <reused>
    PP <rid> <reached> <prompt> <ms>
    T <rid> <id>[,<id>...]
    DONE <rid> <generated> <prompt> <prompt_ms> <decode_ms> <finish> <reused> [key=value ...]
    STATS key=value ...
    ERR <rid|-> <message>

`ERR -` is about the engine, not one request, and the engine exits after it.

`DONE`'s pairs are the request's counters: `queue_ms` from its GEN to its admission, `steps` the decode steps and
draft rounds that fed it one of its own tokens, `drafts` and `accepted` the drafts verified and accepted for it,
and `lookup_drafts` and `lookup_accepted` the part of those from its own tokens (ADR-0041). `STATS` asks for the
engine's counters since it started, and is answered with a `STATS` line: the core's (`engine_stats` in
`include/engine_api.h`, field for field) and the scheduler's (DESIGN.md, section 7). It is answered awake or asleep.

`SLEEP` gives the cards back (ADR-0032). The engine refuses its queued requests and every later `GEN` with
`ERR <rid>`, lets the running ones go on for up to `drain_ms` (default 0), ends the rest with finish `cancel`, frees
its memory on the cards and answers `ASLEEP` with the time from the command. `WAKE` loads the cards again and answers
`INFO` and `READY`, as at the start. `QUIT` ends the engine, awake or asleep.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from enum import StrEnum
from typing import ClassVar

TOP_K_MAX = 64
"""The largest top-k the cards sample from (Strata's limit, kept: the cards exchange top-k candidates)."""


class ProtocolError(ValueError):
    """A line that does not parse."""


class Finish(StrEnum):
    LENGTH = "length"
    STOP = "stop"
    CANCEL = "cancel"


@dataclass(frozen=True, slots=True)
class Settings:
    """A request's sampling settings. Defaults are the model's (spec/MODEL_IO.md, section 2.3)."""

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    penalty_last_n: int = 0
    penalty_repeat: float = 1.0
    penalty_freq: float = 0.0
    penalty_present: float = 0.0
    seed: int = 0

    DEFAULT_PENALTY_LAST_N: ClassVar[int] = 64

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ProtocolError(f"{f.name} must be finite, not {value}")
        if self.temperature < 0:
            raise ProtocolError(f"temperature must be >= 0, not {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ProtocolError(f"top_p must be in (0, 1], not {self.top_p}")
        if not 1 <= self.top_k <= TOP_K_MAX:
            raise ProtocolError(f"top_k must be in 1..{TOP_K_MAX}, not {self.top_k}")
        if not 0 <= self.min_p <= 1:
            raise ProtocolError(f"min_p must be in [0, 1], not {self.min_p}")
        if self.penalty_last_n < 0:
            raise ProtocolError(f"penalty_last_n must be >= 0, not {self.penalty_last_n}")
        if self.penalty_repeat <= 0:
            raise ProtocolError(f"penalty_repeat must be > 0, not {self.penalty_repeat}")
        if not 0 <= self.seed < 2**64:
            raise ProtocolError(f"seed must fit in 64 bits, not {self.seed}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0 or self.top_k == 1

    @property
    def penalised(self) -> bool:
        return self.penalty_repeat != 1.0 or self.penalty_freq != 0.0 or self.penalty_present != 0.0

    def with_defaults(self) -> Settings:
        """Strata's rule: a penalty with no window looks back 64 tokens."""
        if self.penalised and self.penalty_last_n == 0:
            return replace(self, penalty_last_n=self.DEFAULT_PENALTY_LAST_N)
        return self

    def encode(self) -> list[str]:
        """The fields that differ from the defaults, as key=value."""
        default = Settings()
        out: list[str] = []
        for f in fields(self):
            value = getattr(self, f.name)
            if value != getattr(default, f.name):
                out.append(f"{f.name}={_number(value)}")
        return out

    @classmethod
    def decode(cls, pairs: list[str]) -> Settings:
        """Unknown keys are skipped, as Strata's engine does, so an older engine reads a newer front end's lines."""
        types = {f.name: f.type for f in fields(cls)}
        values: dict[str, int | float] = {}
        for pair in pairs:
            key, sep, text = pair.partition("=")
            if not sep:
                raise ProtocolError(f"setting without '=': {pair!r}")
            kind = types.get(key)
            if kind is None:
                continue
            try:
                values[key] = int(text) if kind == "int" else float(text)
            except ValueError:
                raise ProtocolError(f"bad value for {key}: {text!r}") from None
        return cls(**values)  # pyright: ignore[reportArgumentType]


def _number(value: float) -> str:
    return repr(value) if isinstance(value, float) else str(value)


def _ids(text: str) -> tuple[int, ...]:
    if not text:
        raise ProtocolError("empty token list")
    try:
        ids = tuple(int(part) for part in text.split(","))
    except ValueError:
        raise ProtocolError(f"bad token list: {text[:40]!r}") from None
    if any(i < 0 for i in ids):
        raise ProtocolError("negative token id")
    return ids


def _int(text: str, name: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise ProtocolError(f"bad {name}: {text!r}") from None
    if value < 0:
        raise ProtocolError(f"negative {name}: {value}")
    return value


def _float(text: str, name: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ProtocolError(f"bad {name}: {text!r}") from None


def _one_line(text: str) -> str:
    return " ".join(text.split())


# Front end to engine.


@dataclass(frozen=True, slots=True)
class Gen:
    rid: int
    max_new: int
    prompt: tuple[int, ...]
    settings: Settings = field(default_factory=Settings)

    def encode(self) -> str:
        return " ".join(
            ["GEN", str(self.rid), str(self.max_new), *self.settings.encode(), ",".join(map(str, self.prompt))]
        )


@dataclass(frozen=True, slots=True)
class Img:
    """An image, or a pair of a video's frames, of request rid's prompt (docs/spec/MODEL_VISION.md): its block of
    grid_h * grid_w / 4 tokens starts at prompt index `first`; the file at `path` holds grid_h x grid_w patches of
    `frames` x 768 bytes in the encoder's order. `frames` is 1 for a still image, 2 for a frame pair."""

    rid: int
    first: int
    grid_h: int
    grid_w: int
    path: str
    frames: int = 1

    def encode(self) -> str:
        if not self.path or any(c.isspace() for c in self.path):
            raise ProtocolError(f"IMG path must be one word: {self.path!r}")
        return f"IMG {self.rid} {self.first} {self.grid_h} {self.grid_w} {self.frames} {self.path}"


@dataclass(frozen=True, slots=True)
class Stop:
    rid: int

    def encode(self) -> str:
        return f"STOP {self.rid}"


@dataclass(frozen=True, slots=True)
class Stats:
    """Asks for the engine's counters: answered with a StatsReport."""

    def encode(self) -> str:
        return "STATS"


@dataclass(frozen=True, slots=True)
class Sleep:
    drain_ms: int = 0

    def encode(self) -> str:
        return f"SLEEP {self.drain_ms}" if self.drain_ms else "SLEEP"


@dataclass(frozen=True, slots=True)
class Wake:
    def encode(self) -> str:
        return "WAKE"


@dataclass(frozen=True, slots=True)
class Quit:
    def encode(self) -> str:
        return "QUIT"


type Command = Img | Gen | Stop | Stats | Sleep | Wake | Quit


def parse_command(line: str) -> Command:
    parts = line.split()
    if not parts:
        raise ProtocolError("empty line")
    match parts:
        case ["GEN", rid, max_new, *rest] if rest:
            return Gen(_int(rid, "rid"), _int(max_new, "max_new"), _ids(rest[-1]), Settings.decode(rest[:-1]))
        case ["IMG", rid, first, grid_h, grid_w, frames, path]:
            return Img(
                _int(rid, "rid"),
                _int(first, "first"),
                _int(grid_h, "grid_h"),
                _int(grid_w, "grid_w"),
                path,
                _int(frames, "frames"),
            )
        case ["STOP", rid]:
            return Stop(_int(rid, "rid"))
        case ["STATS"]:
            return Stats()
        case ["SLEEP"]:
            return Sleep()
        case ["SLEEP", drain_ms]:
            return Sleep(_int(drain_ms, "drain_ms"))
        case ["WAKE"]:
            return Wake()
        case ["QUIT"]:
            return Quit()
        case _:
            raise ProtocolError(f"unknown command: {line[:60]!r}")


# Engine to front end.


@dataclass(frozen=True, slots=True)
class Info:
    facts: dict[str, str]

    def encode(self) -> str:
        for key, value in self.facts.items():
            if not key or any(c.isspace() or c == "=" for c in key) or any(c.isspace() for c in value):
                raise ProtocolError(f"INFO pair has a space or '=' in the wrong place: {key!r}={value!r}")
        return " ".join(["INFO", *(f"{k}={v}" for k, v in self.facts.items())])


@dataclass(frozen=True, slots=True)
class Ready:
    max_context: int
    slots: int

    def encode(self) -> str:
        return f"READY {self.max_context} {self.slots}"


@dataclass(frozen=True, slots=True)
class Asleep:
    """The engine holds nothing on the cards; `ms` is how long giving them back took."""

    ms: float

    def encode(self) -> str:
        return f"ASLEEP {self.ms:.3f}"


@dataclass(frozen=True, slots=True)
class Resume:
    """`reused` prompt tokens come from a shared prefix or a checkpoint, not from computing them again."""

    rid: int
    reused: int

    def encode(self) -> str:
        return f"RESUME {self.rid} {self.reused}"


@dataclass(frozen=True, slots=True)
class Progress:
    """Prompt progress, one line per prompt chunk; also the request's heartbeat."""

    rid: int
    reached: int
    prompt: int
    ms: float

    def encode(self) -> str:
        return f"PP {self.rid} {self.reached} {self.prompt} {self.ms:.3f}"


@dataclass(frozen=True, slots=True)
class Tokens:
    rid: int
    ids: tuple[int, ...]

    def encode(self) -> str:
        return f"T {self.rid} {','.join(map(str, self.ids))}"


@dataclass(frozen=True, slots=True)
class Done:
    rid: int
    generated: int
    prompt: int
    prompt_ms: float
    decode_ms: float
    finish: Finish
    reused: int = 0
    queue_ms: float = 0.0
    """From the GEN to the request's admission into a slot."""
    steps: int = 0
    """Decode steps and draft rounds that fed the request one of its own tokens."""
    drafts: int = 0
    """Drafts verified for it, and accepted; of them, those from its own tokens (ADR-0041)."""
    accepted: int = 0
    lookup_drafts: int = 0
    lookup_accepted: int = 0

    COUNTERS: ClassVar[tuple[str, ...]] = (
        "queue_ms",
        "steps",
        "drafts",
        "accepted",
        "lookup_drafts",
        "lookup_accepted",
    )
    """The key=value pairs, in order."""

    def encode(self) -> str:
        pairs = " ".join(f"{k}={_number(round(getattr(self, k), 3))}" for k in self.COUNTERS)
        return (
            f"DONE {self.rid} {self.generated} {self.prompt} {self.prompt_ms:.3f} {self.decode_ms:.3f} "
            f"{self.finish} {self.reused} {pairs}"
        )

    @classmethod
    def counters(cls, pairs: list[str]) -> dict[str, int | float]:
        """A DONE line's pairs as the fields they set; a key not known is skipped, as GEN's are."""
        out: dict[str, int | float] = {}
        for pair in pairs:
            key, sep, value = pair.partition("=")
            if not sep:
                raise ProtocolError(f"DONE pair without '=': {pair!r}")
            if key in cls.COUNTERS:
                out[key] = _float(value, key) if key == "queue_ms" else _int(value, key)
        return out


@dataclass(frozen=True, slots=True)
class StatsReport:
    """The engine's counters since it started, as key=value pairs (the answer to Stats)."""

    counts: dict[str, str]

    def encode(self) -> str:
        for key, value in self.counts.items():
            if not key or any(c.isspace() or c == "=" for c in key) or any(c.isspace() for c in value):
                raise ProtocolError(f"STATS pair has a space or '=' in the wrong place: {key!r}={value!r}")
        return " ".join(["STATS", *(f"{k}={v}" for k, v in self.counts.items())])


@dataclass(frozen=True, slots=True)
class Error:
    """`rid` None is an error of the engine itself; the engine exits after sending it."""

    rid: int | None
    message: str

    def encode(self) -> str:
        return f"ERR {'-' if self.rid is None else self.rid} {_one_line(self.message)}"


type Event = Info | Ready | Asleep | Resume | Progress | Tokens | Done | StatsReport | Error


def parse_event(line: str) -> Event:
    parts = line.split()
    if not parts:
        raise ProtocolError("empty line")
    match parts:
        case ["INFO", *pairs]:
            return Info(_pairs(pairs, "INFO"))
        case ["STATS", *pairs]:
            return StatsReport(_pairs(pairs, "STATS"))
        case ["READY", max_context, slots]:
            return Ready(_int(max_context, "max_context"), _int(slots, "slots"))
        case ["ASLEEP", ms]:
            return Asleep(_float(ms, "ms"))
        case ["RESUME", rid, reused]:
            return Resume(_int(rid, "rid"), _int(reused, "reused"))
        case ["PP", rid, reached, prompt, ms]:
            return Progress(_int(rid, "rid"), _int(reached, "reached"), _int(prompt, "prompt"), _float(ms, "ms"))
        case ["T", rid, ids]:
            return Tokens(_int(rid, "rid"), _ids(ids))
        case ["DONE", rid, generated, prompt, prompt_ms, decode_ms, finish, *rest]:
            try:
                why = Finish(finish)
            except ValueError:
                raise ProtocolError(f"bad finish: {finish!r}") from None
            reused = _int(rest[0], "reused") if rest and "=" not in rest[0] else 0
            return Done(
                _int(rid, "rid"),
                _int(generated, "generated"),
                _int(prompt, "prompt"),
                _float(prompt_ms, "prompt_ms"),
                _float(decode_ms, "decode_ms"),
                why,
                reused,
                **Done.counters(rest[1:] if rest and "=" not in rest[0] else rest),  # pyright: ignore[reportArgumentType]
            )
        case ["ERR", rid, *words]:
            return Error(None if rid == "-" else _int(rid, "rid"), " ".join(words))
        case _:
            raise ProtocolError(f"unknown event: {line[:60]!r}")


def _pairs(pairs: list[str], line: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ProtocolError(f"{line} pair without '=': {pair!r}")
        out[key] = value
    return out


def tokens(text: str) -> int:
    """A token count as the engine's and the front end's flags take it: 524288, 512k or 512K (k = 1,024)."""
    t = text.strip().lower()
    n = int(t[:-1]) * 1024 if t.endswith("k") else int(t)
    if n < 0:
        raise ValueError(f"{text}: a negative count")
    return n
