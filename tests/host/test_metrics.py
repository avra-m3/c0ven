"""The ratios `/metrics` reads from the engine's counters."""

from __future__ import annotations

from frontend.metrics import derive, numbers

REPORT = {
    "up_ns": "20000000000",
    "plans": "1000",
    "busy_ns": "15000000000",
    "steps": "400",
    "step_rows": "500",
    "step_ns": "4000000000",
    "step_misses": "20000",
    "step_layers_missed": "8000",
    "undrafted_steps": "100",
    "rounds": "500",
    "round_rows": "2000",
    "round_ns": "10000000000",
    "round_misses": "60000",
    "round_layers_missed": "12000",
    "layer_runs": "450",
    "layer_drafts": "1200",
    "layer_accepted": "900",
    "layer_ns": "9000000000",
    "lookup_runs": "50",
    "lookup_drafts": "300",
    "lookup_accepted": "200",
    "lookup_ns": "1000000000",
    "swap_ns": "1500000000",
    "prompt_rows": "40000",
    "prompt_ns": "20000000000",
    "window_rows": "0",
    "window_ns": "0",
    "ngram_rows": "1000",
    "ngram_hits": "900",
    "decode_ms": "11.250",
    "version": "core-abi9",
}


def test_numbers_read_ints_floats_and_leave_the_rest():
    n = numbers(REPORT)
    assert n["plans"] == 1000
    assert n["decode_ms"] == 11.25
    assert n["version"] == "core-abi9"


def test_derive_divides_what_has_a_divisor():
    d = derive(REPORT)
    assert d["step_ms"] == 10.0
    assert d["round_ms"] == 20.0
    assert d["misses_per_row"] == 32.0
    assert d["layers_missed_per_step"] == round(20000 / 900, 4)
    assert d["step_tokens_per_s"] == 125.0
    assert d["layer_tokens_per_s"] == 150.0, "a run gives its one token and the drafts accepted"
    assert d["lookup_tokens_per_s"] == 250.0
    assert d["layer_accepted_per_run"] == 2.0
    assert d["lookup_accepted_per_run"] == 4.0
    assert d["undrafted_share"] == 0.25
    assert d["busy"] == 0.75
    assert d["swap_share"] == 0.1
    assert d["prompt_rows_per_s"] == 2000.0
    assert "window_rows_per_s" not in d, "no windows ran"
    assert d["ngram_hit_rate"] == 0.9


def test_derive_of_nothing_is_empty():
    assert derive({}) == {}
