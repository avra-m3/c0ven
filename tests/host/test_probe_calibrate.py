"""Host tests for tools/probe_calibrate.py (Platt calibration, threshold selection, and c0ven_probe_v1 fitting)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest

from engine.models.qwen4exp.probe import read_probe_budget
from tools import probe_calibrate


def test_auroc_platt_and_threshold_selection():
    scores = np.array([0.05, 0.1, 0.15, 0.2, 0.75, 0.85, 0.9, 0.95], dtype=np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.float64)
    assert probe_calibrate.auroc(scores, labels) == pytest.approx(1.0)
    scale, bias = probe_calibrate.fit_platt(scores, labels)
    assert scale > 0.0
    assert np.isfinite(bias)
    th, f1, fpr = probe_calibrate.select_threshold(scores, labels, max_fpr=0.05)
    assert 0.2 < th <= 0.75
    assert f1 == pytest.approx(1.0)
    assert fpr == pytest.approx(0.0)


def test_calibrate_cli_writes_valid_calibration_json(tmp_path: Path):
    ds = tmp_path / "eval.jsonl"
    rows = [
        {
            "text": "a",
            "scores": {"agent_safety": 0.05, "hallucination": 0.1},
            "labels": {"agent_safety": 0, "hallucination": 0},
        },
        {
            "text": "b",
            "scores": {"agent_safety": 0.12, "hallucination": 0.2},
            "labels": {"agent_safety": 0, "hallucination": 0},
        },
        {
            "text": "c",
            "scores": {"agent_safety": 0.82, "hallucination": 0.75},
            "labels": {"agent_safety": 1, "hallucination": 1},
        },
        {
            "text": "d",
            "scores": {"agent_safety": 0.91, "hallucination": 0.88},
            "labels": {"agent_safety": 1, "hallucination": 1},
        },
    ]
    ds.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "calibration.json"
    rc = probe_calibrate.main(
        [
            "calibrate",
            str(ds),
            "--out",
            str(out),
            "--probe-model",
            "custom-singprobe",
            "--roll-window",
            "8",
            "--max-fpr",
            "0.05",
        ]
    )
    assert rc == 0
    cal = json.loads(out.read_text())
    assert cal["model_name"] == "custom-singprobe"
    assert cal["roll_window"] == 8
    assert set(cal["thresholds"]) == {"agent_safety", "hallucination"}
    assert set(cal["scale"]) == {"agent_safety", "hallucination"}
    assert set(cal["bias"]) == {"agent_safety", "hallucination"}


def test_fit_linear_drift_writes_loadable_c0ven_probe_v1(tmp_path: Path):
    rng = np.random.default_rng(7)
    n, hidden = 24, 64
    x_before = rng.normal(0.0, 0.5, size=(n, hidden)).astype(np.float32)
    y = np.array([0] * (n // 2) + [1] * (n // 2), dtype=np.float32)
    direction = np.zeros(hidden, dtype=np.float32)
    direction[0] = 3.0
    direction[1] = -2.0
    x_after = x_before + y[:, None] * direction[None, :] + rng.normal(0.0, 0.1, size=(n, hidden)).astype(np.float32)

    npz = tmp_path / "drift.npz"
    np.savez(npz, X_before=x_before, X_after=x_after, y=y)

    out_dir = tmp_path / "drift_probe"
    rc = probe_calibrate.main(
        [
            "fit-linear",
            str(npz),
            "--out-dir",
            str(out_dir),
            "--labels",
            "prompt_injection_drift",
            "--layers",
            "31",
            "--hidden-size",
            str(hidden),
            "--scope",
            "drift",
            "--l2",
            "0.5",
        ]
    )
    assert rc == 0
    cfg = json.loads((out_dir / "config.json").read_text())
    assert cfg["model_type"] == "c0ven_probe_v1"
    assert cfg["base_model_layer_ids"] == [31]
    assert cfg["labels"][0]["name"] == "prompt_injection_drift"
    assert cfg["labels"][0]["scope"] == "drift"

    # Inspect safetensors header and verify BF16 weights separate the classes cleanly.
    raw = (out_dir / "model.safetensors").read_bytes()
    (hdr_len,) = struct.unpack("<Q", raw[:8])
    hdr = json.loads(raw[8 : 8 + hdr_len].decode())
    assert hdr["linear.weight"]["dtype"] == "BF16"
    assert hdr["linear.weight"]["shape"] == [1, hidden]
    assert hdr["linear.bias"]["dtype"] == "F32"
    assert hdr["linear.bias"]["shape"] == [1]

    w_off0, w_off1 = hdr["linear.weight"]["data_offsets"]
    b_off0, b_off1 = hdr["linear.bias"]["data_offsets"]
    payload = raw[8 + hdr_len :]
    w_u16 = np.frombuffer(payload[w_off0:w_off1], dtype="<u2")
    w_f32 = (w_u16.astype("<u4") << 16).view("<f4")
    b_f32 = float(np.frombuffer(payload[b_off0:b_off1], dtype="<f4")[0])
    logits = (x_after - x_before) @ w_f32 + b_f32
    assert float(np.mean(logits[y > 0.5])) > float(np.mean(logits[y <= 0.5])) + 2.0

    budget = read_probe_budget(out_dir)
    assert budget.slot_kv_bytes == 0
    assert budget.fixed_bytes > 0
