"""deepseek41's memory plan: three tiers at 2 cards, 1 slot and 16K, the arena from what the host has."""

from __future__ import annotations

import pytest

from engine import fit
from engine.api.mock import CARD_MEMORY
from engine.host import Host
from engine.memory_plan import GIB, PlanError, PlanOptions, plan
from engine.models.deepseek41 import numbers
from engine.models.deepseek41.memory import model_memory, staging_ring_bytes
from engine.models.deepseek41.setup import memory_plan
from engine.models.deepseek41.tensors import from_tsv
from engine.models.qwen4exp.memory import model_memory as qwen_memory
from engine.models.qwen4exp.tensors import from_tsv as qwen_table

MIB = 2**20
TABLE = from_tsv()


def mem_available(gib: int) -> PlanOptions:
    """The options from a host with that much available, as the engine process builds them."""
    host = Host(
        mem_total=92 * GIB,
        mem_available=gib * GIB,
        swap_total=0,
        cgroup_memory_max=None,
        cgroup_swap_max=None,
        memlock=None,
        cpus=32,
        cpu_quota=None,
        physical_cores=16,
        cpu_flags=frozenset(),
        cpu_model="Test CPU",
    )
    return PlanOptions(slots=1, context=16_384, cards=2, host_memory=host.memory_limit, usable_vram=CARD_MEMORY)


def test_the_model_memory():
    m = model_memory(TABLE)
    assert m.weights == (3_267_502_528, 3_267_502_528)
    assert m.weights_per_card == 3_267_502_528
    assert m.experts == 15_360
    assert m.layer_expert_bytes[0] == 6_912_000
    assert len(m.layer_expert_bytes) == 40
    assert m.ssd_tier
    assert m.engram_cache == 152 * MIB
    assert m.ssd_staging_ring == staging_ring_bytes(TABLE) == 2 * 48 * (8_663_040 + 3 * 4096)
    assert m.routes_per_row == 6
    assert m.host_headroom == numbers.HOST_HEADROOM == 5 * GIB
    assert m.prompt_staging_per_card == m.prompt_host_ring == 0
    assert m.checkpoint_bytes == m.state_per_card == 5_767_168  # 5.5 MiB: the window and compressor rings
    assert m.kv_cell_per_card + m.indexer_cell_per_card == 3_200


def test_the_vram_tier_near_the_estimate():
    """ADR-0080 estimates about 133 experts a layer at layers 0-19 and 128 at 20-39. A layer holds whole slot pairs."""
    p = memory_plan(TABLE, mem_available(64))
    t = p.tiers
    assert t is not None
    assert p.free_per_card / GIB == pytest.approx(18.03, abs=0.005)
    assert sum(t.vram[:20]) / 20 == pytest.approx(133.2)
    assert sum(t.vram[20:]) / 20 == pytest.approx(128.1)
    assert p.experts_cached == sum(t.vram) == 5_226
    assert p.card_layer_experts[0] == p.card_layer_experts[1] == t.vram  # the cards' weights are equal
    # Every layer holds the same bytes, to within two slot pairs of its size for the carry between layers.
    held = [n * b for n, b in zip(t.vram, t.layer_expert_bytes, strict=True)]
    assert max(held) - min(held) < 2 * 2 * max(t.layer_expert_bytes)
    # n / 2 part-A and n / 2 part-B slots on each card: exactly n / 2 experts' bytes, so even counts fit both cards.
    assert all(n % 2 == 0 for n in t.vram)
    on_card = sum(n // 2 * b for n, b in zip(t.vram, t.layer_expert_bytes, strict=True))
    assert on_card <= min(p.card_free)
    assert min(p.card_free) - on_card < 2 * max(t.layer_expert_bytes)


def test_three_tiers_with_the_arena_from_mem_available():
    p = memory_plan(TABLE, mem_available(64))
    t = p.tiers
    assert t is not None
    outside = (
        p.checkpoint_bytes + p.options.host_working + p.engram_cache_bytes + p.ssd_staging_bytes + numbers.HOST_HEADROOM
    )
    room = 64 * GIB - outside
    assert room - 40 * max(t.layer_expert_bytes) < p.arena_bytes <= room
    assert p.arena_bytes == t.bytes_of(t.arena)
    assert p.host_bytes + numbers.HOST_HEADROOM <= 64 * GIB
    assert all(v + a + s == 384 for v, a, s in zip(t.vram, t.arena, t.ssd, strict=True))
    assert t.bytes_of(t.vram) + t.bytes_of(t.arena) + t.bytes_of(t.ssd) == 114_555_617_280
    rows = dict(p.rows())
    assert rows["VRAM tier, experts a layer"].startswith("112-140, ")
    assert rows["arena tier, experts a layer"].startswith("172-214, ")
    assert rows["SSD tier, experts a layer"].startswith("30-100, ")
    assert rows["host headroom"].endswith("(5.00 GiB)")


def test_the_arena_follows_what_the_host_has():
    more = memory_plan(TABLE, mem_available(68))
    less = memory_plan(TABLE, mem_available(60))
    assert more.arena_bytes - less.arena_bytes == pytest.approx(8 * GIB, abs=40 * 13 * MIB)
    assert more.tiers is not None
    assert less.tiers is not None
    assert more.tiers.vram == less.tiers.vram


def test_a_host_with_no_room_for_an_arena_is_still_a_plan():
    p = memory_plan(TABLE, mem_available(8))
    t = p.tiers
    assert t is not None
    assert p.arena_bytes == 0
    assert all(s == 384 - v for s, v in zip(t.ssd, t.vram, strict=True))


def test_a_host_with_room_for_every_expert_leaves_none_on_the_ssd():
    p = memory_plan(TABLE, mem_available(384))
    t = p.tiers
    assert t is not None
    assert p.experts_total == 15_360
    assert all(n % 2 == 0 for n in t.vram)
    assert all(a == 384 - v for a, v in zip(t.arena, t.vram, strict=True))
    assert not any(t.ssd)


def test_the_fit_report_has_the_tiers_at_uniform_routing():
    p = memory_plan(TABLE, mem_available(64))
    t = p.tiers
    assert t is not None
    report = fit.info(fit.tier_rows(p))
    assert report["fit_tier_vram"].startswith("-/112-140_a_layer,")
    reads, size = t.ssd_per_token()
    assert 30 < reads < 40  # about a seventh of the 240 routings a token, with no profile yet
    assert 0.2 * GIB < size < 0.3 * GIB
    assert "uniform_routing" in report["fit_ssd_token"]


def test_each_slot_costs_its_rows_and_rings():
    one = memory_plan(TABLE, mem_available(64))
    options = PlanOptions(slots=2, context=16_384, cards=2, host_memory=64 * GIB, usable_vram=CARD_MEMORY)
    two = memory_plan(TABLE, options)
    assert two.free_per_card == one.free_per_card - (16_384 * 3_200 + 5_767_168)
    assert two.reserve_per_sequence == one.reserve_per_sequence == 16_384 * 3_200 + 5_767_168


def test_the_models_limits():
    p = memory_plan(TABLE, PlanOptions(usable_vram=CARD_MEMORY))
    assert p.options.context == numbers.CONTEXT
    assert p.draft_per_card == 0
    assert not p.options.draft
    assert p.streamed_kv_bytes == 0
    with pytest.raises(PlanError, match="two cards"):
        memory_plan(TABLE, PlanOptions(cards=1, usable_vram=CARD_MEMORY))
    assert memory_plan(TABLE, PlanOptions(ngram_in_ram=True, usable_vram=CARD_MEMORY)) == p


def test_qwen4exps_plan_is_unchanged():
    """With this package imported. tests/host/test_memory_tiers.py holds the whole plan's literals."""
    p = plan(qwen_memory(qwen_table()), PlanOptions(usable_vram=CARD_MEMORY))
    assert (p.weights_per_card, p.experts_cached, p.arena_bytes) == (2_341_860_608, 15_706, 18_151_568_000)
    assert p.tiers is None
