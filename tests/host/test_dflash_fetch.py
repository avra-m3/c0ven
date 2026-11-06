"""tools/dflash_fetch.py's dflash-config.txt: the form core/models/qwen4exp/dflash.cpp, read_df_config, reads."""

from __future__ import annotations

from tools.dflash_fetch import config_text


def test_a_line_a_value_lists_on_one_line():
    text = config_text({"mask_token_id": 248_077, "rms_norm_eps": 1e-6, "target_layer_ids": [5, 15, 25, 35, 45]})
    assert text == "mask_token_id 248077\nrms_norm_eps 1e-06\ntarget_layer_ids 5 15 25 35 45\n"
