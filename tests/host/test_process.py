"""The engine process over its pipe, with the mock behind it."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TextIO, cast

import pytest

import protocol
from engine import fit
from engine.api.interface import StepPlan
from engine.api.mock import MockEngine
from engine.api.reference import ReferenceEngine
from engine.process import SwapWatch, architecture, cache_dir, make_engine, model_module, read_cost, serve
from engine.scheduler import SchedulerOptions, every_16k
from engine.timing import CostModel
from tests.helpers import open_config, write_gguf


def run_engine(lines: list[str], *args: str) -> list[protocol.Event]:
    """Send the lines, read until each request has ended, then QUIT. (QUIT, or the pipe closing, ends the engine at
    once, as when the front end goes away.)"""
    argv = [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", *args]
    p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert p.stdin is not None
    assert p.stdout is not None
    for line in lines:
        p.stdin.write(line + "\n")
    p.stdin.flush()
    events: list[protocol.Event] = []
    ended = 0
    while ended < len(lines):
        line = p.stdout.readline()
        assert line, "the engine closed its output"
        events.append(protocol.parse_event(line))
        ended += isinstance(events[-1], (protocol.Done, protocol.Error))
    p.stdin.write("QUIT\n")
    p.stdin.close()
    assert p.wait(timeout=30) == 0
    return events


def test_info_ready_and_requests():
    events = run_engine(
        [protocol.Gen(1, 3, (1, 2, 3)).encode(), protocol.Gen(2, 2, (4, 5)).encode()],
        "--slots",
        "2",
        "--context",
        "1024",
    )
    assert isinstance(events[0], protocol.Info)
    assert events[0].facts["slots"] == "2"
    assert events[1] == protocol.Ready(1024, 2)
    done = {e.rid: e for e in events if isinstance(e, protocol.Done)}
    assert sorted(done) == [1, 2]
    tokens = {
        rid: [t for e in events if isinstance(e, protocol.Tokens) and e.rid == rid for t in e.ids] for rid in done
    }
    assert [len(tokens[1]), len(tokens[2])] == [done[1].generated, done[2].generated]
    assert all(d.generated <= 3 for d in done.values())


def test_a_request_too_long_is_refused_and_others_go_on():
    events = run_engine([protocol.Gen(1, 5000, (1,)).encode(), protocol.Gen(2, 1, (7,)).encode()], "--context", "512")
    assert protocol.Error(1, "prompt (1) plus max_new (5000) exceeds the context (512)") in events
    assert any(isinstance(e, protocol.Done) and e.rid == 2 for e in events)


def test_a_plan_that_does_not_fit_is_refused():
    p = subprocess.run(
        [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", "--slots", "64"],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert p.returncode == 1
    assert p.stdout.startswith("ERR - memory plan:")


def test_a_plan_over_ram_max_is_refused_with_options_that_fit():
    p = subprocess.run(
        [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", "--ram-max", "15"],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert p.returncode == 1
    assert p.stdout.startswith("ERR - memory plan:")
    assert "--ram-max allows" in p.stdout
    assert "--context" in p.stdout


def test_card_memory_lowers_the_cards_memory_and_never_raises_it():
    from engine.process import card_memory  # noqa: PLC0415

    assert card_memory([], None) is None
    assert card_memory([], 10) == 10 * 2**30
    assert card_memory([24 * 2**30, 23 * 2**30], None) == 23 * 2**30
    assert card_memory([24 * 2**30], 11.5) == round(11.5 * 2**30)
    assert card_memory([12 * 2**30], 16) == 12 * 2**30


def test_the_core_leaves_free_what_card_memory_takes_off():
    from engine.api.interface import CORE_RESERVE  # noqa: PLC0415
    from engine.process import card_reserve  # noqa: PLC0415

    assert card_reserve([24 * 2**30], None) == 0
    assert card_reserve([], 10) == 0
    assert card_reserve([12 * 2**30], 16) == 0
    assert card_reserve([24 * 2**30, 24 * 2**30], 20) == CORE_RESERVE + 4 * 2**30
    assert card_reserve([24 * 2**30, 23 * 2**30], 20) == CORE_RESERVE + 4 * 2**30


def test_a_card_memory_too_small_is_refused_with_what_fits():
    p = subprocess.run(
        [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", "--card-memory", "4"],
        input="",
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert p.returncode == 1
    assert p.stdout.startswith("ERR - memory plan:")
    assert "each has 4.00 GiB" in p.stdout


class Pipe:
    """The engine process, one line at a time."""

    def __init__(self, *args: str) -> None:
        argv = [sys.executable, "-m", "engine", "--backend", "mock", "--arch", "qwen4exp", *args]
        self.p = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)

    def send(self, command: protocol.Command) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(command.encode() + "\n")
        self.p.stdin.flush()

    def until(self, kind: type[protocol.Event]) -> list[protocol.Event]:
        """The events up to and with the first of kind."""
        assert self.p.stdout is not None
        out: list[protocol.Event] = []
        while not out or not isinstance(out[-1], kind):
            line = self.p.stdout.readline()
            assert line, "the engine closed its output"
            out.append(protocol.parse_event(line))
        return out

    def quit(self) -> int:
        self.send(protocol.Quit())
        return self.p.wait(timeout=30)


def test_asleep_the_engine_refuses_requests_and_after_wake_it_serves_again():
    e = Pipe("--context", "1024")
    first = e.until(protocol.Ready)[0]
    assert isinstance(first, protocol.Info)
    e.send(protocol.Gen(1, 4, (1, 2, 3)))
    assert isinstance(e.until(protocol.Done)[-1], protocol.Done)
    e.send(protocol.Sleep())
    assert isinstance(e.until(protocol.Asleep)[-1], protocol.Asleep)
    e.send(protocol.Gen(2, 4, (1, 2, 3)))
    assert e.until(protocol.Error)[-1] == protocol.Error(2, "the engine is asleep")
    e.send(protocol.Wake())
    woke = e.until(protocol.Ready)
    info = woke[0]
    assert isinstance(info, protocol.Info)
    assert info.facts.keys() == first.facts.keys()
    assert woke[-1] == protocol.Ready(1024, 1)
    e.send(protocol.Gen(3, 4, (1, 2, 3)))
    done = e.until(protocol.Done)[-1]
    assert isinstance(done, protocol.Done)
    assert done.rid == 3
    assert e.quit() == 0


def test_images_sent_while_asleep_go_with_their_refused_request():
    """An IMG whose file is missing, then its GEN, while asleep: the GEN is refused for the sleep, and the image does
    not wait for a later request of the same id."""
    e = Pipe("--context", "1024")
    e.until(protocol.Ready)
    e.send(protocol.Sleep())
    e.until(protocol.Asleep)
    e.send(protocol.Img(2, 0, 2, 2, "/nonexistent/image"))
    e.send(protocol.Gen(2, 4, (1, 2, 3)))
    assert e.until(protocol.Error)[-1] == protocol.Error(2, "the engine is asleep")
    e.send(protocol.Wake())
    e.until(protocol.Ready)
    e.send(protocol.Gen(2, 4, (1, 2, 3)))
    e.send(protocol.Gen(3, 4, (1, 2, 3)))
    events = e.until(protocol.Done)
    assert not [x for x in events if isinstance(x, protocol.Error)]
    done = events[-1]
    assert isinstance(done, protocol.Done)
    assert done.rid == 2
    assert e.quit() == 0


def test_sleep_cancels_the_running_request_and_refuses_the_queued_one():
    """One slot: request 1 runs (the mock taking 11 ms a step), request 2 waits for the slot."""
    e = Pipe("--context", "8192", "--realtime")
    e.until(protocol.Ready)
    e.send(protocol.Gen(1, 4000, (1, 2, 3)))
    e.send(protocol.Gen(2, 4, (4, 5)))
    e.until(protocol.Tokens)
    e.send(protocol.Sleep(drain_ms=50))
    events = e.until(protocol.Asleep)
    assert protocol.Error(2, "the engine is going to sleep") in events
    (done,) = [x for x in events if isinstance(x, protocol.Done)]
    assert done.rid == 1
    assert done.finish is protocol.Finish.CANCEL
    assert done.generated < 4000
    asleep = events[-1]
    assert isinstance(asleep, protocol.Asleep)
    assert asleep.ms >= 50
    assert e.quit() == 0


def test_stats_are_answered_awake_and_asleep():
    e = Pipe("--context", "1024")
    e.until(protocol.Ready)
    e.send(protocol.Gen(1, 3, (1, 2, 3)))
    e.until(protocol.Done)
    e.send(protocol.Stats())
    report = e.until(protocol.StatsReport)[-1]
    assert isinstance(report, protocol.StatsReport)
    assert int(report.counts["plans"]) > 0
    assert int(report.counts["steps"]) >= 2
    assert report.counts["writing"] == "0"
    assert int(report.counts["major_faults"]) >= 0
    e.send(protocol.Sleep())
    e.until(protocol.Asleep)
    e.send(protocol.Stats())
    asleep = e.until(protocol.StatsReport)[-1]
    assert isinstance(asleep, protocol.StatsReport)
    assert asleep.counts["plans"] == report.counts["plans"]
    assert e.quit() == 0


def test_the_load_facts_are_in_the_info_line():
    e = Pipe()
    info = e.until(protocol.Ready)[0]
    assert isinstance(info, protocol.Info)
    assert float(info.facts["load_s"]) >= 0
    assert {"pswpout", "compact_stall"} <= info.facts.keys()
    assert e.quit() == 0


class Recording(MockEngine):
    def __init__(self) -> None:
        super().__init__()
        self.plans: list[StepPlan] = []

    def submit(self, plan: StepPlan) -> None:
        self.plans.append(plan)
        super().submit(plan)


def lines_apart(lines: list[str], apart_s: float, quit_after_s: float) -> Iterator[str]:
    """A front end writing its lines one by one, then QUIT."""
    for line in lines:
        time.sleep(apart_s)
        yield line + "\n"
    time.sleep(quit_after_s)
    yield protocol.Quit().encode() + "\n"


def test_requests_sent_together_read_their_prompts_in_one_plan():
    engine = Recording()
    gens = [protocol.Gen(rid, 4, tuple(range(3, 11))).encode() for rid in range(1, 5)]
    stdin = cast("TextIO", lines_apart(gens, apart_s=0.0002, quit_after_s=0.2))
    status = serve(
        engine,
        open_config(slots=4),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=stdin,
        stdout=io.StringIO(),
    )
    assert status == 0
    first = next(p for p in engine.plans if p.prompt)
    assert sorted(span.slot for span in first.prompt) == [0, 1, 2, 3]


def test_the_swap_watch_counts_from_its_first_read(tmp_path: Path):
    file = tmp_path / "vmstat"
    file.write_text("nr_free_pages 9\npswpout 10\ncompact_stall 2\n")
    watch = SwapWatch(file)
    assert watch.pages is None
    assert watch.poll()
    assert watch.pages == 0
    file.write_text("nr_free_pages 9\npswpout 13\ncompact_stall 2\n")
    assert watch.poll()
    assert watch.pages == 3


def test_the_swap_watch_ends_on_a_host_without_the_file(tmp_path: Path):
    watch = SwapWatch(tmp_path / "none", interval_s=3600)
    assert not watch.poll()
    watch.start()
    watch.stop()
    assert watch.pages is None


def wait_for_pages(watch: SwapWatch, pages: int) -> None:
    deadline = time.monotonic() + 5
    while watch.pages != pages:
        assert time.monotonic() < deadline, f"the watch stopped at {watch.pages} pages"
        time.sleep(0.001)


def test_the_swap_watch_outlasts_a_read_that_finds_nothing(tmp_path: Path):
    """Once the host has shown a count, a failed read is not a host without one: the next read goes on from it."""
    file = tmp_path / "vmstat"
    file.write_text("pswpout 10\n")
    watch = SwapWatch(file, interval_s=0.001)
    watch.start()
    try:
        wait_for_pages(watch, 0)
        file.unlink()
        time.sleep(0.05)
        assert watch.pages == 0
        file.write_text("pswpout 17\n")
        wait_for_pages(watch, 7)
    finally:
        watch.stop()


def stats_after(file: Path | None) -> Iterator[str]:
    """A front end that sees the host swap, asks for STATS once the watch can have read it, then quits."""
    time.sleep(0.2)
    if file is not None:
        file.write_text("pswpout 15\n")
    time.sleep(0.2)
    yield protocol.Stats().encode() + "\n"
    time.sleep(0.2)
    yield protocol.Quit().encode() + "\n"


def test_swap_out_is_in_the_counters(tmp_path: Path):
    file = tmp_path / "vmstat"
    file.write_text("pswpout 10\n")
    out = io.StringIO()
    status = serve(
        MockEngine(),
        open_config(),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=cast("TextIO", stats_after(file)),
        stdout=out,
        swap_watch=SwapWatch(file, interval_s=0.01),
    )
    assert status == 0
    report = next(line for line in out.getvalue().splitlines() if line.startswith("STATS"))
    assert "swapped_out_pages=5" in report.split()


def test_no_swap_counter_without_the_file(tmp_path: Path):
    out = io.StringIO()
    status = serve(
        MockEngine(),
        open_config(),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=cast("TextIO", stats_after(None)),
        stdout=out,
        swap_watch=SwapWatch(tmp_path / "none", interval_s=0.01),
    )
    assert status == 0
    report = next(line for line in out.getvalue().splitlines() if line.startswith("STATS"))
    assert "swapped_out_pages" not in report


def test_the_fit_report_is_in_the_info_line():
    rows = [fit.Row("arena", "arena", "", "1.00GiB", "0.98GiB"), fit.Row("swap", "swap", "8.00GiB,cgroup:-")]
    out = io.StringIO()
    status = serve(
        MockEngine(),
        open_config(),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=io.StringIO("QUIT\n"),
        stdout=out,
        fit_report=lambda _facts: (rows, []),
    )
    assert status == 0
    info = protocol.parse_event(out.getvalue().splitlines()[0])
    assert isinstance(info, protocol.Info)
    assert info.facts["fit_arena"] == "-/1.00GiB/0.98GiB"
    assert info.facts["fit_swap"] == "8.00GiB,cgroup:-/-/-"


def test_the_core_far_from_the_plan_is_noted():
    from engine.memory_plan import PlanOptions, plan  # noqa: PLC0415
    from engine.models.qwen4exp.memory import model_memory  # noqa: PLC0415
    from engine.models.qwen4exp.tensors import from_tsv  # noqa: PLC0415

    p = plan(model_memory(from_tsv()), PlanOptions())
    assert fit.card_notes(p, p.experts_per_layer + 3) == []
    assert "the core caches" in fit.card_notes(p, p.experts_per_layer + 60)[0]
    assert fit.card_notes(p, p.experts_per_layer, p.working_per_card + 2**20) == []
    assert "working memory" in fit.card_notes(p, p.experts_per_layer, p.working_per_card + 2**30)[0]


def test_the_whole_fit_report_after_a_load(tmp_path: Path):
    import dataclasses  # noqa: PLC0415

    from engine import host  # noqa: PLC0415
    from engine.memory_plan import PlanOptions, plan  # noqa: PLC0415
    from engine.models.qwen4exp.memory import model_memory  # noqa: PLC0415
    from engine.models.qwen4exp.tensors import from_tsv  # noqa: PLC0415
    from engine.process import report  # noqa: PLC0415
    from tests.helpers import open_config as config  # noqa: PLC0415

    p = plan(model_memory(from_tsv()), PlanOptions())
    card = host.Card("B60", "0000:03:00.0", 24 * 2**30, 0xE211, (20 << 22) | (1 << 14), 20, 5, 8, -1)
    facts = dataclasses.replace(MockEngine().open(config()), host_read_bytes_per_s=40e9)
    rows, _ = report(host.read(tmp_path, affinity={0, 1}), [card, card], p, 2, facts)
    info = fit.info(rows)
    assert info["fit_host_read"] == "-/-/40.0GB/s"
    assert info["fit_cpu"].endswith("/2_workers")


def test_the_cache_dir_is_read_under_its_new_name_then_its_old(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ENGINE_CACHE_DIR", raising=False)
    monkeypatch.delenv("ENGINE_MTP_CACHE_DIR", raising=False)
    assert cache_dir() is None
    monkeypatch.setenv("ENGINE_MTP_CACHE_DIR", "/old")
    assert cache_dir() == "/old"
    monkeypatch.setenv("ENGINE_CACHE_DIR", "/new")
    assert cache_dir() == "/new"


def serve_lines(lines: list[str], cost_file: Path) -> int:
    stdin = cast("TextIO", lines_apart(lines, apart_s=0.01, quit_after_s=0.3))
    return serve(
        MockEngine(),
        open_config(),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=stdin,
        stdout=io.StringIO(),
        cost_file=cost_file,
    )


def test_the_cost_model_is_kept_at_exit_and_read_at_the_next_start(tmp_path: Path):
    file = tmp_path / "cache" / "cost-x.json"
    assert serve_lines([protocol.Gen(1, 4, tuple(range(3, 300))).encode()], file) == 0
    kept = read_cost(file)
    assert kept is not None
    assert kept.to_dict()["s"] != CostModel.initial().to_dict()["s"], "the mock's prompt chunk was fitted"
    seen = CostModel.initial(decode_ms=3.0, row_ms=0.2, chunk_ms=50.0)
    file.write_text(json.dumps(seen.to_dict()))
    assert serve_lines([], file) == 0, "no plan: the model read at start is written back as it came"
    back = read_cost(file)
    assert back is not None
    assert (back.row_ms, back.chunk_ms) == (0.2, 50.0)


def test_the_cost_model_is_kept_at_a_sleep(tmp_path: Path):
    file = tmp_path / "cost-x.json"
    lines = [protocol.Gen(1, 4, tuple(range(3, 40))).encode(), protocol.Sleep().encode()]
    stdin = cast("TextIO", lines_apart(lines, apart_s=0.2, quit_after_s=0.3))
    serve(MockEngine(), open_config(), rule=every_16k, options=SchedulerOptions(), stdin=stdin, stdout=io.StringIO())
    assert not file.exists(), "no file without the cache directory"
    stdin = cast("TextIO", lines_apart(lines, apart_s=0.2, quit_after_s=0.3))
    status = serve(
        MockEngine(),
        open_config(),
        rule=every_16k,
        options=SchedulerOptions(),
        stdin=stdin,
        stdout=io.StringIO(),
        cost_file=file,
    )
    assert status == 0
    assert read_cost(file) is not None


def test_a_cost_file_that_does_not_read_is_left_for_the_estimates(tmp_path: Path):
    assert read_cost(None) is None
    assert read_cost(tmp_path / "none.json") is None
    bad = tmp_path / "cost-x.json"
    bad.write_text("{")
    assert read_cost(bad) is None
    bad.write_text(json.dumps({"row_ms": 1.0}))
    assert read_cost(bad) is None


def test_the_architecture_is_read_from_the_first_shard(tmp_path: Path):
    write_gguf(tmp_path / "m-00001-of-00006.gguf", "testarch")
    (tmp_path / "m-00004-of-00006.gguf").write_bytes(b"")
    assert architecture(tmp_path / "m-00004-of-00006.gguf") == "testarch"
    assert architecture(tmp_path / "m-00001-of-00006.gguf") == "testarch"


def test_an_architecture_without_a_setup_module_is_refused():
    with pytest.raises(SystemExit, match="'testarch' is not supported"):
        model_module("testarch")


def test_the_reference_backend_refuses_an_architecture_it_has_no_records_of():
    with pytest.raises(SystemExit, match="records of qwen4exp only, not of testarch"):
        make_engine("reference", arch="testarch")
    assert isinstance(make_engine("reference", arch="qwen4exp"), ReferenceEngine)
