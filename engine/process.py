"""The engine process: the protocol on stdin and stdout, the scheduler between (DESIGN.md, sections 1 and 2).

The front end starts it as its child. One thread reads commands, one writes events, so the loop never blocks on
I/O (vLLM's engine core does the same). The loop admits, plans and polls through the scheduler. On `SLEEP` it
drains the scheduler and gives the cards back; on `WAKE` it loads them again with a new scheduler (ADR-0032). `STATS`
is answered from the counters the core and the scheduler keep, awake or asleep.

    python -m engine --model /models/<file>-00001-of-00002.gguf --slots 4 --context 262144
    python -m engine --backend mock --arch <arch> --slots 4 --context 32768
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import os
import queue
import sys
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import TextIO, cast

import gguf

import protocol
from engine.api.interface import IMAGE_PATCH_BYTES, Engine, EngineError, EngineFacts, Image, OpenConfig
from engine.memory_plan import MemoryPlan, PlanError, PlanOptions
from engine.scheduler import CheckpointRule, Scheduler, SchedulerOptions

log = logging.getLogger("engine")

BACKENDS = ("native", "mock", "reference")

GATHER_S = 0.002
"""After a GEN reaches an idle engine, the loop waits this long for the next command before it plans, so requests
sent together read their prompts in one plan."""

VMSTAT_KEYS = ("pswpout", "compact_stall")
"""What a load is checked by: pages the host swapped out, and the times an allocation stalled to compact memory
(ADR-0032)."""


def vmstat(path: Path = Path("/proc/vmstat")) -> dict[str, int]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return {}
    out: dict[str, int] = {}
    for line in lines:
        key, _, value = line.partition(" ")
        if key in VMSTAT_KEYS:
            out[key] = int(value)
    return out


class Load:
    """One open or wake, timed, with the host's swapping and compaction across it."""

    def __init__(self) -> None:
        self.t0 = time.monotonic()
        self.before = vmstat()

    def facts(self) -> dict[str, str]:
        after = vmstat()
        out = {"load_s": f"{time.monotonic() - self.t0:.2f}"}
        for key in VMSTAT_KEYS:
            if key in self.before and key in after:
                out[key] = str(after[key] - self.before[key])
        return out


class State(Enum):
    AWAKE = "awake"
    DRAINING = "draining"
    """SLEEP came: no request is admitted, and the running ones end, or are cancelled at the deadline."""
    ASLEEP = "asleep"


def architecture(model: Path) -> str:
    reader = gguf.GGUFReader(model)
    field = reader.get_field("general.architecture")
    if field is None:
        raise SystemExit(f"{model} has no general.architecture")
    return str(field.contents())


def model_module(arch: str) -> ModuleType:
    try:
        return importlib.import_module(f"engine.models.{arch}.setup")
    except ModuleNotFoundError:
        raise SystemExit(f"architecture {arch!r} is not supported") from None


def make_engine(backend: str, realtime: bool = False) -> Engine:
    match backend:
        case "native":
            from engine.api.native import NativeEngine  # noqa: PLC0415

            return NativeEngine()
        case "mock":
            from engine.api.mock import MockEngine  # noqa: PLC0415

            return MockEngine(realtime=realtime)
        case "reference":
            from engine.api.reference import ReferenceEngine  # noqa: PLC0415

            return ReferenceEngine()
        case _:
            raise SystemExit(f"unknown backend {backend!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m engine", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--backend", choices=BACKENDS, default="native")
    p.add_argument("--model", type=Path, help="the GGUF file (any shard); needed by the native backend")
    p.add_argument("--arch", help="the architecture, when there is no model file (mock, reference)")
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--context", type=int, default=262_144)
    p.add_argument(
        "--checkpoints",
        type=int,
        default=8,
        help="slot checkpoints in pinned host memory, each a slot's running state (118,038,528 B)",
    )
    p.add_argument(
        "--kv-tokens",
        type=protocol.tokens,
        default=0,
        help="the KV pool the slots share, in tokens (512k = 524,288), at least --context; default: slots x context",
    )
    p.add_argument("--table-cache", type=Path, help="a directory that keeps the model's tensor table between starts")
    p.add_argument(
        "--swapout-limit-mib",
        type=float,
        help="fail a load or wake that makes the host swap out more than this (ERR -); default: report it only",
    )
    p.add_argument("--deterministic", action="store_true", help="ADR-0016's switches, for level-6 tests")
    p.add_argument(
        "--draft-dir",
        type=Path,
        help="the drafter's tensors, DFlash 2's or the MTP layer's; default: dflash/ beside the model file when it"
        " holds dflash-manifest.json, else mtp/ when it holds mtp-manifest.json",
    )
    p.add_argument("--no-draft", action="store_true", help="decode without the draft layer")
    p.add_argument(
        "--vision",
        type=Path,
        help="the image encoder's file; default: the one mmproj-*.gguf beside the model file, when there is one",
    )
    p.add_argument("--no-vision", action="store_true", help="read no images")
    p.add_argument("--prompt-budget-ms", type=float, default=SchedulerOptions().prompt_budget_ms)
    p.add_argument("--decode-share", type=float, default=SchedulerOptions().decode_share)
    p.add_argument("--chunk-rows", type=int, default=SchedulerOptions().chunk_rows)
    p.add_argument("--realtime", action="store_true", help="the mock sleeps for the step times it reports")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args(argv)
    if args.backend == "native" and args.model is None:
        p.error("the native backend needs --model")
    if args.model is None and args.arch is None:
        p.error("give --model or --arch")
    return args


def _reader(stream: TextIO, commands: queue.Queue[protocol.Command]) -> None:
    """A line that does not parse is the front end's bug, not a request's: it is logged and skipped."""
    for line in stream:
        if not line.strip():
            continue
        try:
            commands.put(protocol.parse_command(line))
        except protocol.ProtocolError as exc:
            log.error("bad command: %s", exc)
    commands.put(protocol.Quit())


def vision_file(model: Path) -> Path | None:
    """The image encoder beside the model file: the one `mmproj-*.gguf` there, or None."""
    found = sorted(model.parent.glob("mmproj-*.gguf"))
    return found[0] if len(found) == 1 else None


def read_image(img: protocol.Img) -> Image:
    """An IMG line's image or frame pair, its pixels read from the front end's file now."""
    if img.frames not in (1, 2):
        raise ValueError(f"an image has 1 frame or 2, not {img.frames}")
    size = img.grid_h * img.grid_w * img.frames * IMAGE_PATCH_BYTES
    with Path(img.path).open("rb") as f:
        pixels = f.read(size + 1)
    if len(pixels) != size:
        raise ValueError(f"{img.path} holds {len(pixels)} bytes, not {size}")
    image = Image(img.first, img.grid_h, img.grid_w, pixels, img.frames)
    image.check()
    return image


def _writer(stream: TextIO, events: queue.Queue[protocol.Event | None]) -> None:
    while (event := events.get()) is not None:
        stream.write(event.encode() + "\n")
        if events.empty():
            stream.flush()
    stream.flush()


def serve(
    engine: Engine,
    config: OpenConfig,
    *,
    rule: CheckpointRule,
    options: SchedulerOptions,
    stdin: TextIO,
    stdout: TextIO,
    info: Callable[[], dict[str, str]] = dict,
    swapout_limit_mib: float | None = None,
) -> int:
    events: queue.Queue[protocol.Event | None] = queue.Queue()
    commands: queue.Queue[protocol.Command] = queue.Queue()
    writer = threading.Thread(target=_writer, args=(stdout, events), name="protocol-writer", daemon=True)
    writer.start()

    def loaded(facts: EngineFacts, load: Load) -> str | None:
        """INFO and READY for a load; or, past the swap-out limit, the reason to fail."""
        measured = load.facts()
        pages = int(measured.get("pswpout", "0"))
        if swapout_limit_mib is not None and pages * 4096 > swapout_limit_mib * 2**20:
            return f"the host swapped out {pages * 4096 / 2**20:.0f} MiB while loading, over {swapout_limit_mib:g}"
        facts_info = {
            "version": facts.version,
            "cards": str(facts.cards),
            "slots": str(facts.slots),
            "max_context": str(facts.max_context),
            "pages": str(facts.pages),
            "checkpoints": str(facts.checkpoints),
            "experts_cached": str(facts.experts_cached),
            "pid": str(os.getpid()),
            "timing_fd": str(facts.timing_fd),
            "image_tokens_max": str(facts.image_tokens_max),
            **measured,
            **info(),
        }
        events.put(protocol.Info(facts_info))
        events.put(protocol.Ready(facts.max_context, facts.slots))
        return None

    load = Load()
    try:
        facts = engine.open(config)
    except EngineError as exc:
        events.put(protocol.Error(None, f"open failed: {exc}"))
        events.put(None)
        writer.join()
        return 1
    if (failure := loaded(facts, load)) is not None:
        engine.close()
        events.put(protocol.Error(None, failure))
        events.put(None)
        writer.join()
        return 1
    reader = threading.Thread(target=_reader, args=(stdin, commands), name="protocol-reader", daemon=True)
    reader.start()
    scheduler = Scheduler(engine, facts, options, rule)
    images: dict[int, list[Image]] = {}
    """Each request's images from its IMG lines, until its GEN."""
    bad_images: dict[int, str] = {}
    state = State.AWAKE
    sleep_t0 = 0.0
    deadline = 0.0
    cancelled = False
    wake_after = False
    status = 0
    try:
        while True:
            try:
                command = commands.get(timeout=0.5) if scheduler.idle else commands.get_nowait()
            except queue.Empty:
                command = None
            quit_now = False
            wake_now = False
            gather = scheduler.idle
            while command is not None:
                match command:
                    case protocol.Img(rid=rid):
                        try:
                            images.setdefault(rid, []).append(read_image(command))
                        except (OSError, ValueError) as exc:
                            bad_images.setdefault(rid, f"image {len(images.get(rid, []))}: {exc}")
                    case protocol.Gen(rid=rid) if state is not State.AWAKE:
                        images.pop(rid, None)
                        bad_images.pop(rid, None)
                        why = "asleep" if state is State.ASLEEP else "going to sleep"
                        events.put(protocol.Error(rid, f"the engine is {why}"))
                    case protocol.Gen(rid=rid):
                        ims = tuple(images.pop(rid, ()))
                        if (problem := bad_images.pop(rid, None)) is not None:
                            events.put(protocol.Error(rid, problem))
                        else:
                            for e in scheduler.add(command, images=ims):
                                events.put(e)
                    case protocol.Stop(rid=rid):
                        for e in scheduler.cancel(rid):
                            events.put(e)
                    case protocol.Stats():
                        events.put(protocol.StatsReport(scheduler.report()))
                    case protocol.Sleep(drain_ms=drain_ms) if state is State.AWAKE:
                        state, sleep_t0, cancelled, wake_after = State.DRAINING, time.monotonic(), False, False
                        deadline = sleep_t0 + drain_ms / 1e3
                        for e in scheduler.refuse_waiting("the engine is going to sleep"):
                            events.put(e)
                    case protocol.Sleep():
                        if state is State.ASLEEP:
                            events.put(protocol.Asleep(0.0))
                    case protocol.Wake():
                        match state:
                            case State.ASLEEP:
                                wake_now = True
                            case State.DRAINING:
                                wake_after = True
                            case State.AWAKE:
                                events.put(protocol.Ready(facts.max_context, facts.slots))
                    case protocol.Quit():
                        quit_now = True
                wait = gather and isinstance(command, protocol.Gen)
                try:
                    command = commands.get(timeout=GATHER_S) if wait else commands.get_nowait()
                except queue.Empty:
                    command = None
            if quit_now:
                break
            if state is State.DRAINING:
                if not scheduler.idle and not cancelled and time.monotonic() >= deadline:
                    cancelled = True
                    for rid in list(scheduler.by_rid):
                        for e in scheduler.cancel(rid):
                            events.put(e)
                if scheduler.idle:
                    engine.sleep()
                    state = State.ASLEEP
                    events.put(protocol.Asleep((time.monotonic() - sleep_t0) * 1e3))
                    wake_now = wake_after
            if wake_now:
                load = Load()
                facts = engine.wake()
                if (failure := loaded(facts, load)) is not None:
                    raise RuntimeError(failure)
                scheduler = Scheduler(engine, facts, options, rule)
                state = State.AWAKE
            if not scheduler.idle:
                for e in scheduler.tick(0.002):
                    events.put(e)
    except Exception as exc:  # every failure ends the process with ERR -, so the front end restarts it
        log.exception("engine failed")
        events.put(protocol.Error(None, str(exc)))
        status = 1
    finally:
        try:
            engine.close()
        except EngineError:
            log.exception("close failed")
        events.put(None)
        writer.join()
    return status


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    arch = args.arch or architecture(args.model)  # the front end passes --arch, so the GGUF is read once
    model = model_module(arch)
    options = PlanOptions(
        slots=args.slots,
        context=args.context,
        checkpoints=args.checkpoints,
        kv_tokens=args.kv_tokens,
    )
    native = args.backend == "native"
    draft: Path | None = None
    if native and not args.no_draft and args.model is not None:
        draft = args.draft_dir or cast("Callable[[Path], Path | None]", model.draft_dir)(args.model)
    try:
        config, plan = cast(
            "tuple[OpenConfig, MemoryPlan]",
            model.open_config(args.model if native else None, options, args.deterministic, draft, args.table_cache),
        )
    except PlanError as exc:
        print(protocol.Error(None, f"memory plan: {exc}").encode(), flush=True)
        return 1
    for warning in plan.warnings:
        log.warning("memory plan: %s", warning)
    vision = None if args.no_vision or not native or args.model is None else args.vision or vision_file(args.model)
    if vision is not None:
        config = dataclasses.replace(config, vision_path=str(vision))
    sched = SchedulerOptions(
        prompt_budget_ms=args.prompt_budget_ms, decode_share=args.decode_share, chunk_rows=args.chunk_rows
    )
    rule = cast("CheckpointRule", model.checkpoint_positions)
    return serve(
        make_engine(args.backend, args.realtime),
        config,
        rule=rule,
        options=sched,
        stdin=sys.stdin,
        stdout=sys.stdout,
        info=lambda: {"backend": args.backend, "arch": arch, "experts_planned": str(plan.experts_cached)},
        swapout_limit_mib=args.swapout_limit_mib,
    )
