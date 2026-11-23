from __future__ import annotations

import pytest

from engine.models.inkling2_loop import split, tensors


@pytest.mark.parametrize("cards", [1, 2, 3, 4, 5, 6, 7, 8])
def test_card_slices_partition_heads_ff_and_vocab(cards: int):
    table = tensors.from_tsv()
    slices = split.card_slices(table, cards)
    assert len(slices) == cards
    assert sum(s.kv_n for s in slices) == table.head_count_kv
    assert sum(s.q_n for s in slices) == table.head_count
    assert sum(s.ff_n for s in slices) == table.expert_ff
    assert sum(s.v_n for s in slices) == table.vocab_unpadded
    align = split.ff_alignment(table)
    assert all(s.ff_n > 0 and s.ff_n % align == 0 for s in slices)
    assert all(s.kv_n > 0 and s.q_n == s.kv_n * 3 for s in slices)


def test_card_partition_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="alignment"):
        split.card_partition(10, 2, align=4)
    with pytest.raises(ValueError, match="blocks"):
        split.card_partition(8, 3, align=4)


def test_bytes_per_card_splits_dense_weights():
    table = tensors.from_tsv()
    one = split.bytes_per_card(table, 1)
    two = split.bytes_per_card(table, 2)
    four = split.bytes_per_card(table, 4)
    eight = split.bytes_per_card(table, 8)
    assert len(one) == 1
    assert two[0] == two[1]
    assert two[0] < one[0]
    assert four[0] < two[0]
    assert max(eight) < max(four)


def test_expert_bytes_per_card_scales_across_cards():
    table = tensors.from_tsv()
    one = split.expert_bytes_per_card(table, 1)
    two = split.expert_bytes_per_card(table, 2)
    four = split.expert_bytes_per_card(table, 4)
    assert one[0][0] == split.layer_expert_bytes(table)[0]
    assert two[0][0] * 2 == one[0][0]
    assert four[0][0] * 4 == one[0][0]
