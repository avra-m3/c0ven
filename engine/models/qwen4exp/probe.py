"""Size a streaming hidden-state probe directory for qwen4exp's memory plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

PROBE_FILES = ("config.json", "model.safetensors")
PROMPT_CHUNK = 4352


@dataclass(frozen=True, slots=True)
class ProbeBudget:
    """Device memory on card 0 and per-slot KV ring bytes for a streaming probe."""

    fixed_bytes: int = 0
    slot_kv_bytes: int = 0


def read_probe_budget(path: Path) -> ProbeBudget:
    """Compute card-0 fixed bytes (weights and prompt scratch) and per-slot KV ring bytes from a probe dir."""
    if not all((path / name).is_file() for name in PROBE_FILES):
        raise FileNotFoundError(f"no streaming probe in {path} (expected {', '.join(PROBE_FILES)})")
    cfg = cast("dict[str, Any]", json.loads((path / "config.json").read_text()))
    kind = str(cfg.get("model_type", "sing_probe_attn"))
    embd = int(cfg.get("hidden_size", 2560))
    layers = cfg.get("base_model_layer_ids")
    n_taps = len(cast("list[Any]", layers)) if isinstance(layers, list) and layers else 1
    feat = n_taps * embd

    extra_linear = 0
    cal_path = path / "calibration.json"
    if cal_path.is_file():
        cal = cast("dict[str, Any]", json.loads(cal_path.read_text()))
        extra = cal.get("extra_alarms")
        if isinstance(extra, str) and extra:
            extra_cfg = cast("dict[str, Any]", json.loads((path / extra / "config.json").read_text()))
            extra_linear = len(cast("list[Any]", extra_cfg.get("labels", ())))

    slot_kv = 0
    tail_bytes = 0
    if kind == "sing_probe_attn":
        num_heads = int(cfg.get("num_attention_heads", 8))
        head_dim = int(cfg.get("head_dim", 64))
        window = int(cfg.get("sliding_window", 2048))
        num_labels = int(cfg.get("num_labels", 10))
        q_dim = num_heads * head_dim
        proj_rows = q_dim + 2 * head_dim + extra_linear
        slot_kv = 2 * window * head_dim * 2
        tail_bytes = q_dim * q_dim * 2 + q_dim * 4 + (num_labels * q_dim + num_labels) * 4
    elif kind == "sing_probe_mlp":
        inter = int(cfg.get("intermediate_size", 512))
        num_labels = int(cfg.get("num_labels", 10))
        proj_rows = inter + extra_linear
        tail_bytes = (inter + num_labels * inter + num_labels) * 4
    elif kind == "c0ven_probe_v1":
        n_lin = len(cfg.get("labels", ())) + extra_linear
        proj_rows = max(1, n_lin)
    else:
        raise ValueError(f"unsupported probe model_type: {kind!r}")

    stride = (proj_rows + 31) & ~31
    # Card 0 holds proj_full + proj_tap (BF16) plus prompt scratch pqkv + ptmp (F32).
    fixed = 2 * stride * feat * 2 + 2 * PROMPT_CHUNK * stride * 4 + tail_bytes
    return ProbeBudget(fixed_bytes=fixed, slot_kv_bytes=slot_kv)
