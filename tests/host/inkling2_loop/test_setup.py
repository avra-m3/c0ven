from __future__ import annotations

from pathlib import Path

from engine.api import abi
from engine.memory_plan import GIB
from engine.models.inkling2_loop import setup
from tests.helpers import plan_options


def test_draft_and_vision_discovery(tmp_path: Path):
    model = tmp_path / "Inkling-2-Mini-Q8_0.gguf"
    model.write_bytes(b"")
    assert setup.draft_dir(model) is None
    assert setup.vision_file(model) is None

    dspark = tmp_path / "dspark-Inkling-2-Mini-Q8_0.gguf"
    mmproj = tmp_path / "mmproj-Inkling-2-Mini-BF16.gguf"
    dspark.write_bytes(b"")
    mmproj.write_bytes(b"")
    assert setup.draft_dir(model) == dspark
    assert setup.vision_file(model) == mmproj


def test_open_config_passes_max_cards_beyond_two():
    config, mp = setup.open_config(
        None,
        plan_options(cards=4, context=65_536, usable_vram=24 * GIB),
    )
    assert mp.cards == 4
    assert config.max_cards == 4
    assert config.vocab_size == 200_058
    assert config.eos_ids == (200_006,)

    _, mp_clamped = setup.open_config(
        None,
        plan_options(cards=abi.MAX_CARDS + 4, context=65_536, usable_vram=24 * GIB),
    )
    assert mp_clamped.cards == 8
