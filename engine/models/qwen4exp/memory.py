"""qwen4exp's numbers for the memory plan, from its tensor table and split rules (MEMORY_BUDGET.md)."""

from __future__ import annotations

from typing import cast

from engine.api.interface import SplitKind
from engine.memory_plan import ModelMemory
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.draft import draft_kv_cell, draft_per_card, read_shapes
from engine.models.qwen4exp.split import bytes_per_card, split_of
from engine.models.qwen4exp.tensors import Category, LayerKind, TensorTable


def _q8(values: int) -> int:
    n, nbytes = numbers.KV_BLOCK
    return values // n * nbytes


def kv_cell_bytes(table: TensorTable) -> int:
    """K and V of one attention layer, per token: the KV heads' keys and values."""
    a = table.architecture
    heads = _int(table, f"{a}.attention.head_count_kv")
    return _q8(heads * (_int(table, f"{a}.attention.key_length") + _int(table, f"{a}.attention.value_length")))


def indexer_cell_bytes(table: TensorTable) -> int:
    """The indexer's keys of one attention layer, per token: one key for its heads."""
    return _q8(_int(table, f"{table.architecture}.attention.indexer.key_length"))


def deltanet_state_bytes(table: TensorTable) -> int:
    """One DeltaNet layer's state (spec/MODEL_LAYER.md, 2.7): the conv's last `conv_kernel - 1` rows of q, k and v,
    and the recurrence, a state_size square for each value head."""
    a = table.architecture
    inner, state = _int(table, f"{a}.ssm.inner_size"), _int(table, f"{a}.ssm.state_size")
    conv_rows = (_int(table, f"{a}.ssm.conv_kernel") - 1) * (inner + 2 * _int(table, f"{a}.ssm.group_count") * state)
    recurrence = state * inner
    return (conv_rows + recurrence) * numbers.STATE_VALUE_BYTES


def ple_history_bytes(table: TensorTable) -> int:
    """The PLE conv history of the layers that have one (spec/MODEL_IO.md): `conv_kernel - 1` steps of each n-gram
    position, a hyper-connection's width each."""
    a = table.architecture
    width = _int(table, f"{a}.hyper_connection.count") * _int(table, f"{a}.embedding_length")
    steps = (_int(table, f"{a}.ple.conv_kernel") - 1) * _int(table, f"{a}.ple.ngram_size")
    layers = len(cast("list[int]", table.field(f"{a}.ple.layers")))
    return layers * steps * width * numbers.STATE_VALUE_BYTES


def _int(table: TensorTable, key: str) -> int:
    return int(cast("int", table.field(key)))


def model_memory(
    table: TensorTable, cards: int = 2, draft_shapes: dict[str, tuple[int, ...]] | None = None
) -> ModelMemory:
    """The sizes on one card or each of two: one card holds every weight and every head whole. draft_shapes: the
    drafter's tensor shapes (draft.read_shapes); None: the committed MTP layer's."""
    if cards not in (1, 2):
        raise ValueError(f"the engine runs on one or two cards, not {cards}")
    expert_total = table.bytes_of(Category.EXPERTS)
    experts = table.layers * table.experts_per_layer
    kinds = table.layer_kinds()
    attention = kinds.count(LayerKind.ATTENTION)
    kv_cell = attention * kv_cell_bytes(table)
    indexer_cell = attention * indexer_cell_bytes(table)
    state = kinds.count(LayerKind.DELTANET) * deltanet_state_bytes(table) + ple_history_bytes(table)
    shapes = draft_shapes if draft_shapes is not None else read_shapes()
    draft_kv = numbers.DRAFT_WINDOW * _q8(draft_kv_cell(shapes))
    return ModelMemory(
        cards=cards,
        weights_per_card=sum(bytes_per_card(t, cards) for t in table.tensors),
        draft_per_card=draft_per_card(shapes, cards),
        working_per_card=numbers.WORKING_PER_CARD,
        layers=table.layers,
        experts=experts,
        expert_bytes=expert_total // experts,
        kv_cell_per_card=kv_cell // cards,
        indexer_cell_per_card=indexer_cell // cards,
        state_per_card=state // cards,
        draft_kv_per_card=draft_kv // cards,
        resident_cells=numbers.RESIDENT_CELLS,
        kv_cell_host=kv_cell,
        draft_kv_host=draft_kv,
        checkpoint_bytes=state,
        max_context=table.context_length,
        ngram_bytes=sum(t.nbytes for t in table.tensors if split_of(t.name) is SplitKind.FILE),
    )
