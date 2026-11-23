"""Memory accounting for inkling2_loop across 1 to ENGINE_MAX_CARDS cards."""

from __future__ import annotations

from pathlib import Path

from engine.memory_plan import MIB, ModelMemory
from engine.models.inkling2_loop.draft import (
    draft_kv_cell_per_card,
    draft_kv_per_card,
    draft_weights_per_card,
    read_dspark,
)
from engine.models.inkling2_loop.split import bytes_per_card, card_slices, layer_expert_bytes
from engine.models.inkling2_loop.tensors import TensorTable

FP16 = 2
F32 = 4
SWA_RING_EXTRA = 64
"""Extra SWA ring cells past `sliding_window` so speculative verify windows never overwrite live keys."""
WORKING_PER_CARD = 640 * MIB
"""*Estimate*: activations, attention workspace, scratch and peer exchange buffers on each card."""


def swa_ring_cells(table: TensorTable) -> int:
    return table.sliding_window + SWA_RING_EXTRA


def shortconv_state_bytes(table: TensorTable, kv_n: int) -> int:
    """One sequence's shortconv history across all logical slots on a card holding `kv_n` KV heads, bytes."""
    hist = table.shortconv_kernel - 1
    channels = 2 * kv_n * table.head_dim + 2 * table.embedding_length
    return table.logical_layers * hist * channels * F32


def swa_state_bytes(table: TensorTable, kv_n: int) -> int:
    """One sequence's fixed SWA FP16 KV rings on a card holding `kv_n` KV heads, bytes."""
    return table.swa_layers * swa_ring_cells(table) * 2 * kv_n * table.head_dim * FP16


def checkpoint_state_bytes(table: TensorTable) -> int:
    """Full fixed state saved per checkpoint (all KV heads' SWA rings + all shortconv history), bytes."""
    hist = table.shortconv_kernel - 1
    conv = table.logical_layers * hist * (2 * table.head_count_kv * table.head_dim + 2 * table.embedding_length) * F32
    swa = table.swa_layers * swa_ring_cells(table) * 2 * table.head_count_kv * table.head_dim * FP16
    return conv + swa


def model_memory(
    table: TensorTable,
    cards: int,
    *,
    draft: Path | None = None,
    vision: Path | None = None,
) -> ModelMemory:
    """Build `ModelMemory` for `table` split across `cards` cards."""
    slices = card_slices(table, cards)
    max_kv = max(sl.kv_n for sl in slices)
    weights = list(bytes_per_card(table, cards))
    if vision is not None and vision.is_file():
        weights[0] += vision.stat().st_size

    draft_w = 0
    draft_kv = 0
    draft_global_cell = 0
    if draft is not None and draft.is_file():
        dinfo = read_dspark(draft)
        draft_w = draft_weights_per_card(dinfo, cards)
        draft_kv = draft_kv_per_card(dinfo, cards)
        draft_global_cell = draft_kv_cell_per_card(dinfo, cards)

    kv_cell = table.global_layers * 2 * max_kv * table.head_dim * FP16 + draft_global_cell
    state_card = swa_state_bytes(table, max_kv) + shortconv_state_bytes(table, max_kv)
    per_layer_exp = layer_expert_bytes(table)
    mean_exp = sum(per_layer_exp) // len(per_layer_exp)

    return ModelMemory(
        cards=cards,
        trained_context=table.context_length,
        max_context=table.context_length,
        weights_per_card=max(weights),
        card_weights=tuple(weights),
        draft_per_card=draft_w,
        working_per_card=WORKING_PER_CARD,
        kv_cell_per_card=kv_cell,
        kv_cell_host=0,
        resident_cells=table.context_length,
        indexer_cell_per_card=0,
        state_per_card=state_card,
        draft_kv_per_card=draft_kv,
        draft_kv_host=0,
        checkpoint_bytes=checkpoint_state_bytes(table),
        ngram_bytes=0,
        experts=table.layers * table.experts_per_layer,
        layers=table.layers,
        expert_bytes=mean_exp,
        layer_expert_bytes=per_layer_exp,
        routes_per_row=table.experts_used,
    )
