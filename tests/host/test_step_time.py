"""The step-time model against the measured steps, and its layout generator's constraints."""

from __future__ import annotations

import math
from dataclasses import replace

import gguf
import pytest

from engine.memory_plan import PlanError, PlanOptions, plan
from engine.models.qwen4exp.memory import model_memory
from engine.models.qwen4exp.steps import measured_terms, model_steps
from engine.models.qwen4exp.tensors import FileTensor, TensorTable, from_tsv
from engine.step_time import (
    Layout,
    LayoutError,
    counted_miss_wait_ms,
    hit_rate,
    layout,
    layouts,
    miss_wait_ms,
    step_time,
)

TABLE = from_tsv()
STEPS = model_steps(TABLE)
MEMORY = model_memory(TABLE)
TERMS = measured_terms()
OPTIONS = PlanOptions()
TODAY = Layout(1, 2, TERMS.cpu_wait_experts, attention_by_heads=True)
"""Two cards in one group with the cache the measured steps ran with."""


def _down_as_iq4_nl(t: FileTensor) -> FileTensor:
    if not (t.name.endswith("ffn_down_exps.weight") and t.type_name == "Q2_0"):
        return t
    kind = gguf.GGMLQuantizationType.IQ4_NL
    values, size = gguf.GGML_QUANT_SIZES[kind]
    return replace(t, type_name=kind.name, ggml_type=int(kind), nbytes=math.prod(t.shape) // values * size)


NL_TABLE = TensorTable(tuple(_down_as_iq4_nl(t) for t in TABLE.tensors))
"""The IQ3_S table with its nine Q2_0 downs as IQ4_NL."""
NL_STEPS = model_steps(NL_TABLE)
NL_MEMORY = model_memory(NL_TABLE)


def test_calibration():
    """F(1) is the measured one-row step less the model's other terms."""
    s = step_time(TERMS, STEPS, TODAY, 1)
    print("\n".join(s.lines()))
    assert TERMS.fixed_ms is not None
    assert 0 < TERMS.fixed_ms < 8
    assert s.step_ms == pytest.approx(TERMS.step_ms)


@pytest.mark.parametrize(("rows", "measured"), [(2, 19.25), (4, 25.46)])
def test_the_measured_steps(rows: int, measured: float):
    """Steps of rows from one request (docs/measurements/MEASUREMENTS.md#2026-10-19-four-rows-a-step)."""
    s = step_time(TERMS, STEPS, TODAY, rows)
    print("\n".join(s.lines()))
    assert s.step_ms == pytest.approx(measured, rel=0.15)


@pytest.mark.parametrize("rows", [1, 2, 3, 4])
def test_two_cards_in_one_group_are_cheapest_for_one_request(rows: int):
    two = layouts(2, STEPS, TERMS, memory=MEMORY, options=OPTIONS)
    assert [str(lay) for lay in two] == ["groups(1, 2)", "groups(2, 1)"]
    times = {str(lay): step_time(TERMS, STEPS, lay, rows).token_ms for lay in two}
    assert min(times, key=times.__getitem__) == "groups(1, 2)"


def test_four_rows_from_four_requests():
    """The core runs the groups in turn, so a token takes the step over its rows; pipelined, groups(2, 1) would take
    half that. The pipelining rule is unchecked, so this also prints the two layouts' times a token."""
    for lay in layouts(2, STEPS, TERMS, memory=MEMORY, options=OPTIONS):
        s = step_time(TERMS, STEPS, lay, 4, requests=4)
        p = step_time(TERMS, STEPS, lay, 4, requests=4, pipelined=True)
        assert s.token_ms == pytest.approx(s.step_ms / 4)
        assert p.token_ms == pytest.approx(s.step_ms / (4 * lay.groups))
        print(f"{lay}: {s.step_ms:.2f} ms a step, {s.token_ms:.2f} ms a token, {p.token_ms:.2f} pipelined")


def test_four_cards_need_downs_in_whole_blocks():
    """Q2_0's 64-value blocks do not divide a quarter of 640 columns; IQ4_NL's 32 do."""
    with pytest.raises(LayoutError, match="Q2_0"):
        layout(1, 4, 4, STEPS, TERMS, memory=MEMORY, options=OPTIONS)
    four = layout(1, 4, 4, NL_STEPS, TERMS, memory=NL_MEMORY, options=OPTIONS)
    pairs = layout(2, 2, 4, STEPS, TERMS, memory=MEMORY, options=OPTIONS)
    assert four.experts_cached == pairs.experts_cached == 512
    assert not four.attention_by_heads
    assert pairs.attention_by_heads


def test_four_cards_in_one_group_save_time():
    """At one row, a quarter of each layer a card. Held at the two-card cache, the saving is the thinner slices' less
    the reduces of four cards, three peer writes each; with the cache four cards hold, the CPU waits for nothing."""
    two = step_time(TERMS, NL_STEPS, TODAY, 1)
    four = layout(1, 4, 4, NL_STEPS, TERMS, memory=NL_MEMORY, options=OPTIONS)
    held = step_time(TERMS, NL_STEPS, four, 1, experts_cached=TODAY.experts_cached)
    curve = step_time(TERMS, NL_STEPS, four, 1)
    print("\n".join(two.lines() + held.lines() + curve.lines()))
    assert 1.9 < two.step_ms - held.step_ms < 2.4
    assert two.step_ms - curve.step_ms > two.step_ms - held.step_ms
    assert curve.miss_ms == 0


def test_six_cards_in_three_pairs():
    """3 groups divide the 48 layers; the layout is rejected only if the memory plan does not fit."""
    try:
        plan(MEMORY, OPTIONS, cards=2)
    except PlanError:
        with pytest.raises(LayoutError):
            layout(3, 2, 6, STEPS, TERMS, memory=MEMORY, options=OPTIONS)
    else:
        assert layout(3, 2, 6, STEPS, TERMS, memory=MEMORY, options=OPTIONS).groups == 3


def test_five_groups_do_not_divide_the_layers():
    with pytest.raises(LayoutError, match="divide"):
        layout(5, 1, 5, STEPS, TERMS, memory=MEMORY, options=OPTIONS)
    assert layouts(5, STEPS, TERMS, memory=MEMORY, options=OPTIONS) == []


def test_one_card():
    assert [str(lay) for lay in layouts(1, STEPS, TERMS, memory=MEMORY, options=OPTIONS)] == ["groups(1, 1)"]


@pytest.mark.parametrize(("groups", "width", "cards"), [(1, 2, 4), (2, 2, 2), (1, 1, 2), (3, 1, 4), (0, 2, 0)])
def test_layouts_that_do_not_use_the_cards(groups: int, width: int, cards: int):
    with pytest.raises(LayoutError, match="does not use"):
        layout(groups, width, cards, STEPS, TERMS, memory=MEMORY, options=OPTIONS)


def test_a_width_with_no_rates():
    with pytest.raises(LayoutError, match="no rates"):
        layout(1, 3, 3, STEPS, TERMS, memory=MEMORY, options=OPTIONS)


@pytest.mark.parametrize("requests", [0, 3])
def test_requests_outside_the_rows(requests: int):
    with pytest.raises(ValueError, match="requests"):
        step_time(TERMS, STEPS, TODAY, 2, requests=requests)


@pytest.mark.parametrize(("slots", "context"), [(1, 32_768), (4, 32_768), (1, 262_144), (4, 262_144)])
def test_the_plan_at_two_cards_is_the_default(slots: int, context: int):
    o = PlanOptions(slots=slots, context=context)
    assert plan(MEMORY, o, cards=2) == plan(MEMORY, o)


def test_four_cards_hold_a_quarter_of_each_expert():
    p = plan(MEMORY, OPTIONS, cards=4)
    quarter = -(-MEMORY.expert_bytes // 4)
    assert p.experts_cached == min(MEMORY.experts, p.free_per_card // quarter)
    assert p.experts_cached > plan(MEMORY, OPTIONS).experts_cached


@pytest.mark.parametrize("rows", [1, 2, 4])
def test_misses_counted_at_the_curves_rate_cost_the_curves_m(rows: int):
    h = hit_rate(TERMS, TERMS.cpu_wait_experts)
    layers, slots = len(STEPS.layers), STEPS.experts_a_token * rows
    counted = counted_miss_wait_ms(TERMS, STEPS, rows, layers * (1 - h**slots), layers * slots * (1 - h))
    assert counted == pytest.approx(miss_wait_ms(TERMS, STEPS, TERMS.cpu_wait_experts, rows))
    assert counted_miss_wait_ms(TERMS, STEPS, rows, 0, 0) == 0
