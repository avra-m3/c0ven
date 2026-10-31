"""The mock refuses the plans the core would refuse, so the scheduler's tests mean something."""

from __future__ import annotations

from dataclasses import replace

from engine.api.interface import EventKind, PageChange, PageOp, PromptSpan, SlotSettings, StepPlan
from engine.api.mock import MockEngine
from engine.api.reference import ReferenceEngine, load_reference
from tests.helpers import open_config

SETTINGS = SlotSettings(0.0, 1.0, 1, 0.0, 0, 1.0, 0.0, 0.0, 0, 4)


def run(engine: MockEngine) -> list[EventKind]:
    kinds: list[EventKind] = []
    for _ in range(20):
        kinds += [e.kind for e in engine.poll()]
    return kinds


def test_a_prompt_row_without_a_page_is_an_error():
    m = MockEngine()
    m.open(open_config())
    m.admit(0, SETTINGS, (5, 6, 7, 8, 9))
    m.submit(StepPlan(1, [PromptSpan(0, 0, 5, True)], [PageChange(PageOp.ASSIGN, 0, 0, 3)]))
    assert EventKind.ERROR in run(m)


def test_a_gap_between_spans_is_an_error():
    m = MockEngine()
    m.open(open_config())
    m.admit(0, SETTINGS, (5, 6, 7, 8, 9))
    m.submit(StepPlan(1, [PromptSpan(0, 2, 3, True)], [PageChange(PageOp.ASSIGN, 0, 0, 3)]))
    assert EventKind.ERROR in run(m)


def test_a_shared_page_read_back_wrong_is_an_error():
    """Two slots given the same page for different tokens: the first one's history no longer reads back."""
    m = MockEngine(lambda _: 7)
    m.open(open_config())
    two = replace(SETTINGS, max_new=2)
    m.admit(0, two, (5, 6))
    m.admit(1, two, (8, 9))
    m.submit(StepPlan(1, [PromptSpan(0, 0, 2, True)], [PageChange(PageOp.ASSIGN, 0, 0, 3)]))
    m.submit(StepPlan(2, [PromptSpan(1, 0, 2, True)], [PageChange(PageOp.ASSIGN, 1, 0, 3)]))
    assert EventKind.ERROR in run(m)


def test_restoring_a_checkpoint_of_another_prompt_is_an_error():
    m = MockEngine()
    m.open(open_config())
    m.admit(0, SETTINGS, (5, 6, 7))
    m.submit(StepPlan(1, [PromptSpan(0, 0, 2, False)], [PageChange(PageOp.ASSIGN, 0, 0, 1)]))
    m.checkpoint(0, 0)
    m.admit(1, SETTINGS, (9, 9, 9))
    m.restore(1, 0)
    kinds = run(m)
    assert EventKind.CHECKPOINT_DONE in kinds
    assert EventKind.ERROR in kinds


def test_the_reference_replays_llama_cpp():
    records = load_reference()
    prompt, expected = next((p, g) for p, g in records.items() if len(p) < 100)
    ref = ReferenceEngine(records)
    ref.open(replace(open_config(context=4096), vocab_size=248_320))
    ref.admit(0, SlotSettings(0.0, 1.0, 1, 0.0, 0, 1.0, 0.0, 0.0, 0, 8), prompt)
    pages = [PageChange(PageOp.ASSIGN, 0, i, i) for i in range(-(-(len(prompt) + 8) // 4))]
    ref.submit(StepPlan(1, [PromptSpan(0, 0, len(prompt), True)], pages))
    tokens: list[int] = []
    for _ in range(20):
        tokens += [e.a for e in ref.poll() if e.kind is EventKind.TOKEN]
    assert tokens == list(expected[:8])


def test_the_reference_refuses_an_unknown_prompt():
    ref = ReferenceEngine({(1, 2): (3,)})
    ref.open(open_config())
    ref.admit(0, SETTINGS, (4, 5))
    ref.submit(StepPlan(1, [PromptSpan(0, 0, 2, True)], [PageChange(PageOp.ASSIGN, 0, 0, 0)]))
    assert EventKind.ERROR in run(ref)


def test_a_span_counts_as_a_chunk_when_it_is_longer_than_the_plans_window_rows():
    chunks: list[int] = []
    for window_rows in (0, 20, 19, 8):
        m = MockEngine()
        m.open(open_config())
        m.admit(0, SETTINGS, tuple(range(5, 25)))
        plan = StepPlan(1, [PromptSpan(0, 0, 20, True)], [PageChange(PageOp.ASSIGN, 0, 0, 5)], window_rows=window_rows)
        m.submit(plan)
        done = [e for _ in range(20) for e in m.poll() if e.kind is EventKind.STEP_DONE]
        chunks.append(done[0].c)
    assert chunks == [0, 0, 1, 1], "0 is 64 rows; a span of 20 rows is a window up to 20"
