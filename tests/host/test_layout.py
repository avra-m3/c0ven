"""The layout chooser, the step log and the load check's terms."""

from __future__ import annotations

from dataclasses import replace

import pytest

from engine.api.interface import StepRecord
from engine.layout import REQUESTS, StepLog, choose, parse
from engine.memory_plan import PlanOptions
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.steps import MEASURED, checked_terms, measured_terms, model_steps, with_check
from engine.models.qwen4exp.tensors import from_tsv
from engine.step_time import Layout, LayoutError, counted_miss_wait_ms, rate, step_time
from tests.helpers import load_check

TABLE = from_tsv()
STEPS = model_steps(TABLE)
MEMORY = model_memory(TABLE)
TERMS = measured_terms()
OPTIONS = PlanOptions()
TENSOR = {(1, 2)}


@pytest.mark.parametrize(
    ("spec", "want"), [("auto", None), ("tensor", (1, 2)), ("layers", (2, 1)), ("groups:2x1", (2, 1))]
)
def test_the_layouts_asked_for(spec: str, want: tuple[int, int] | None):
    assert parse(spec, 2) == want


@pytest.mark.parametrize("spec", ["groups", "groups:2", "groups:axb", "rows", ""])
def test_a_layout_that_does_not_parse(spec: str):
    with pytest.raises(ValueError, match="--layout"):
        parse(spec, 2)


def test_auto_takes_the_layout_the_core_runs_and_costs_them_all():
    c = choose("auto", 2, STEPS, TERMS, memory=MEMORY, options=OPTIONS, runs=TENSOR)
    assert str(c.layout) == "groups(1, 2)"
    assert set(c.costs) == {"groups(1, 2)", "groups(2, 1)"}
    assert all(len(v) == len(REQUESTS) for v in c.costs.values())
    print(c.line())
    assert "a token at 1, 2, 4, 8 requests: groups(1, 2) " in c.line()


def test_auto_takes_the_cheapest_for_one_request_of_those_the_core_runs():
    c = choose("auto", 2, STEPS, TERMS, memory=MEMORY, options=OPTIONS, runs={(1, 2), (2, 1)})
    assert str(c.layout) == "groups(1, 2)"


def test_a_layout_the_core_does_not_run_is_refused():
    with pytest.raises(LayoutError, match="does not run"):
        choose("layers", 2, STEPS, TERMS, memory=MEMORY, options=OPTIONS, runs=TENSOR)
    with pytest.raises(LayoutError, match="does not use"):
        choose("groups:3x1", 2, STEPS, TERMS, memory=MEMORY, options=OPTIONS, runs=TENSOR)
    with pytest.raises(LayoutError, match="core runs"):
        choose("auto", 2, STEPS, TERMS, memory=MEMORY, options=OPTIONS, runs={(4, 4)})


def at(kernel: str, rows: int, i: int) -> float:
    """A rate the record has."""
    v = MEASURED.rates[kernel][rows][i]
    assert v is not None
    return v


def record(plan: int, rows: int, ms: float, misses: int = 0, layers_missed: int = 0) -> StepRecord:
    return StepRecord(
        "decode",
        plan,
        rows,
        int(ms * 1e6),
        misses=misses,
        layers_missed=layers_missed,
        card0_ns=int((ms - 1) * 1e6),
        cpu_wait_ns=200_000,
    )


def test_the_step_log_sets_each_row_counts_steps_against_the_prediction():
    today = Layout(1, 2, TERMS.cpu_wait_experts, attention_by_heads=True)
    log = StepLog(TERMS, STEPS, today, every=8)
    curve = step_time(TERMS, STEPS, today, 1)
    # Steps without a miss: the prediction is the curve's less its M.
    no_m = curve.step_ms - curve.miss_ms
    first = [record(i, 1, no_m) for i in range(5)]
    assert log.feed(first) == []
    assert log.feed(first) == [], "the same steps again are not new"
    steps = [*first, *(record(5 + i, 4, 25.5, misses=120, layers_missed=30) for i in range(3))]
    lines = log.feed(steps)
    assert len(lines) == 2
    print("\n".join(lines))
    assert lines[0].startswith(
        f"groups(1, 2), 1 row, 5 steps: predicted {no_m:.2f} ms at their misses (15.50 at the hit-rate curve's), "
        f"measured {no_m:.2f} (+0.0%)"
    )
    four = step_time(TERMS, STEPS, today, 4)
    m = counted_miss_wait_ms(TERMS, STEPS, 4, 30, 120)
    assert lines[1].startswith(f"groups(1, 2), 4 rows, 3 steps: predicted {four.step_ms - four.miss_ms + m:.2f} ms")
    assert f"CPU wait predicted {m:.2f} ({four.miss_ms:.2f}), measured 0.20" in lines[1]
    assert log.feed(steps) == []


def test_a_check_that_matches_the_record_changes_no_rate():
    same = load_check(
        gate_up={w: {r: at("gate_up IQ3_S", r, i) for r in (1, 2, 4)} for i, w in enumerate((1, 2))},
        down={w: {r: at("down IQ4_NL", r, i) for r in (1, 2, 4)} for i, w in enumerate((1, 2))},
        dense={1: at("attn_qkv", 1, 0), 2: at("attn_qkv", 1, 1)},
    )
    out = with_check(MEASURED, same)
    for k, by in MEASURED.rates.items():
        for r, values in by.items():
            assert out.rates[k][r] == pytest.approx(values), k


def test_a_faster_kernel_moves_its_group_at_the_widths_it_was_timed():
    check = load_check(
        gate_up={
            1: {1: at("gate_up IQ3_S", 1, 0) * 1.1},
            2: {1: at("gate_up IQ3_S", 1, 1) * 1.1, 4: at("gate_up IQ3_S", 4, 1) * 0.9},
        },
        dense={2: at("attn_qkv", 1, 1) * 0.8},
        exchange={1: 4.0, 4: 7.0},
        handoff=20.0,
    )
    out = with_check(MEASURED, check)
    xxs = MEASURED.rates["gate_up IQ3_XXS"]
    assert out.rates["gate_up IQ3_XXS"][1][1] == pytest.approx(at("gate_up IQ3_XXS", 1, 1) * 1.1)
    assert out.rates["gate_up IQ3_XXS"][2][1] == pytest.approx(at("gate_up IQ3_XXS", 2, 1) * 1.1), (
        "2 rows take the one-row change"
    )
    assert out.rates["gate_up IQ3_XXS"][4][1] == pytest.approx(at("gate_up IQ3_XXS", 4, 1) * 0.9)
    assert out.rates["gate_up IQ3_XXS"][1][2:] == xxs[1][2:], "4 and 8 cards were not timed"
    assert out.rates["down IQ4_NL"] == MEASURED.rates["down IQ4_NL"]
    assert out.rates["output"][1][1] == pytest.approx(at("output", 1, 1) * 0.8), "the head is matvec too"
    assert out.rates["output"][1][0] == at("output", 1, 0)
    assert (out.exchange_us, out.handoff_us) == ({1: 4.0, 4: 7.0}, 20.0)
    assert out.fixed_ms is None


def test_a_timed_kernel_without_a_record_enters_alone():
    out = with_check(MEASURED, replace(load_check(gate_up={2: {1: 240.0}}), gate_up_type="IQ4_XS"))
    assert out.rates["gate_up IQ4_XS"] == {1: (None, 240.0, None, None)}
    assert out.rates["gate_up IQ3_S"] == MEASURED.rates["gate_up IQ3_S"]


def test_checked_terms_are_calibrated_on_the_measured_step():
    terms = checked_terms(load_check(gate_up={2: {1: at("gate_up IQ3_S", 1, 1) * 1.2}}))
    today = Layout(1, 2, terms.cpu_wait_experts, attention_by_heads=True)
    assert step_time(terms, STEPS, today, 1).step_ms == pytest.approx(MEASURED.step_ms)
    assert terms.fixed_ms is not None
    assert terms.fixed_ms > (TERMS.fixed_ms or 0), "faster products leave more of the step to F"
    assert rate(terms, "gate_up IQ3_S", 2, 1) == pytest.approx(at("gate_up IQ3_S", 1, 1) * 1.2)
