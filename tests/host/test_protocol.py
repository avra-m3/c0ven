from __future__ import annotations

import pytest

import protocol
from protocol import (
    Asleep,
    Done,
    Error,
    Finish,
    Gen,
    Info,
    Progress,
    ProtocolError,
    Quit,
    Ready,
    Resume,
    Settings,
    Sleep,
    Stats,
    StatsReport,
    Stop,
    Tokens,
    Wake,
    parse_command,
    parse_event,
)


@pytest.mark.parametrize(
    "command",
    [
        Gen(7, 128, (248045, 8678, 198)),
        Gen(1, 1, (0,), Settings(temperature=0.0, top_k=1, seed=2**64 - 1)),
        Gen(2, 5, (1, 2), Settings(penalty_repeat=1.1, penalty_last_n=64, min_p=0.05)),
        Stop(9),
        Stats(),
        Sleep(),
        Sleep(30000),
        Wake(),
        Quit(),
    ],
)
def test_commands_round_trip(command: protocol.Command):
    assert parse_command(command.encode()) == command


@pytest.mark.parametrize(
    "event",
    [
        Info({"slots": "4", "version": "0.1"}),
        Ready(262144, 4),
        Asleep(612.5),
        Resume(3, 2048),
        Progress(3, 2048, 4096, 950.125),
        Tokens(3, (248046,)),
        Tokens(3, (1, 2, 3)),
        Done(3, 128, 66, 9.5, 104.25, Finish.LENGTH, 0),
        Done(4, 1, 2, 0.0, 0.0, Finish.CANCEL, 12),
        Done(5, 40, 66, 9.5, 104.25, Finish.STOP, 0, 12.125, 20, 30, 19, 7, 6),
        StatsReport({"plans": "12", "decode_ms": "11.250"}),
        Error(5, "the prompt is too long"),
        Error(None, "device lost"),
    ],
)
def test_events_round_trip(event: protocol.Event):
    assert parse_event(event.encode()) == event


def test_gen_line_matches_the_documented_form():
    line = Gen(12, 64, (1, 2, 3), Settings(temperature=0.5)).encode()
    assert line == "GEN 12 64 temperature=0.5 1,2,3"


def test_unknown_settings_are_skipped():
    assert parse_command("GEN 1 4 future_key=3 temperature=0.0 5,6") == Gen(1, 4, (5, 6), Settings(temperature=0.0))


def test_done_without_reused_reads_as_zero():
    """An older engine's DONE line, without the last field."""
    assert parse_event("DONE 1 2 3 1.0 2.0 stop") == Done(1, 2, 3, 1.0, 2.0, Finish.STOP, 0)


def test_done_line_matches_the_documented_form():
    line = Done(5, 40, 66, 9.5, 104.25, Finish.STOP, 3, 12.125, 20, 30, 19, 7, 6).encode()
    assert line == (
        "DONE 5 40 66 9.500 104.250 stop 3 "
        "queue_ms=12.125 steps=20 drafts=30 accepted=19 lookup_drafts=7 lookup_accepted=6"
    )


def test_done_counters_may_be_partial_or_unknown():
    assert parse_event("DONE 1 2 3 1.0 2.0 stop 4 steps=2 future=9") == Done(1, 2, 3, 1.0, 2.0, Finish.STOP, 4, steps=2)
    assert parse_event("DONE 1 2 3 1.0 2.0 stop queue_ms=0.5") == Done(1, 2, 3, 1.0, 2.0, Finish.STOP, 0, 0.5)


def test_error_messages_are_one_line():
    assert Error(1, "two\nlines").encode() == "ERR 1 two lines"


@pytest.mark.parametrize(
    "line",
    [
        "",
        "GEN 1 4",
        "GEN 1 4 temperature=0.5",
        "GEN x 4 1,2",
        "GEN 1 4 1,,2",
        "GEN 1 4 -1",
        "GEN 1 4 top_k=0 1",
        "GEN 1 4 top_k=65 1",
        "GEN 1 4 temperature=nan 1",
        "GEN 1 4 top_p=0 1",
        "STOP",
        "HELLO",
    ],
)
def test_bad_commands(line: str):
    with pytest.raises(ProtocolError):
        parse_command(line)


@pytest.mark.parametrize(
    "line",
    [
        "T 1",
        "T 1 a",
        "DONE 1 2 3 1.0 2.0 finished",
        "DONE 1 2 3 1.0 2.0 stop 0 steps",
        "READY 1",
        "PP 1 2 3",
        "STATS x",
        "WAT",
    ],
)
def test_bad_events(line: str):
    with pytest.raises(ProtocolError):
        parse_event(line)


def test_info_refuses_spaces():
    with pytest.raises(ProtocolError):
        Info({"version": "a b"}).encode()


def test_penalty_window_default():
    assert Settings(penalty_repeat=1.1).with_defaults().penalty_last_n == 64
    assert Settings().with_defaults().penalty_last_n == 0


@pytest.mark.parametrize(("text", "n"), [("524288", 524_288), ("512k", 524_288), ("256K", 262_144), ("0", 0)])
def test_token_counts(text: str, n: int):
    assert protocol.tokens(text) == n


@pytest.mark.parametrize("text", ["-1", "1.5k", "k", "lots"])
def test_bad_token_counts(text: str):
    with pytest.raises(ValueError, match=r"negative|invalid literal"):
        protocol.tokens(text)
