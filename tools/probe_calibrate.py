"""Calibrate streaming probe thresholds/Platt parameters and fit `c0ven_probe_v1` linear or drift probes.

Subcommands:
  calibrate   Score a labeled JSONL file against a running server (or precomputed scores) and write calibration.json.
  fit-linear  Fit L2-regularized logistic regression from `.npz` hidden states (or before/after drift pairs) into a
              `c0ven_probe_v1` probe directory (`config.json` + `model.safetensors`).
"""

from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

type FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LabelFit:
    """Calibration metrics and fitted parameters for one probe label."""

    name: str
    samples: int
    positives: int
    auroc: float
    scale: float
    bias: float
    threshold: float
    f1: float
    fpr: float


def _sigmoid(x: FloatArray) -> FloatArray:
    clipped = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(p: FloatArray) -> FloatArray:
    eps = 1e-6
    q = np.clip(p, eps, 1.0 - eps)
    return np.log(q / (1.0 - q))


def auroc(scores: FloatArray, labels: FloatArray) -> float:
    """Area under the ROC curve via rank statistics; 0.5 when only one class is present."""
    pos = labels > 0.5
    n_pos = int(np.sum(pos))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
    rank_sum = float(np.sum(ranks[pos]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fit_platt(probs: FloatArray, labels: FloatArray, l2: float = 1e-2, max_iter: int = 50) -> tuple[float, float]:
    """Fit Platt scaling sigmoid(scale * logit(p) + bias) by Newton-IRLS with small L2 regularization."""
    if len(probs) == 0 or int(np.sum(labels > 0.5)) == 0 or int(np.sum(labels <= 0.5)) == 0:
        return 1.0, 0.0
    x = _logit(probs)
    design = np.column_stack((x, np.ones_like(x)))
    theta = np.array([1.0, 0.0], dtype=np.float64)
    prior = np.array([1.0, 0.0], dtype=np.float64)
    reg = np.diag([l2, l2 * 0.1])
    for _ in range(max_iter):
        pred = _sigmoid(design @ theta)
        w = np.maximum(pred * (1.0 - pred), 1e-9)
        grad = design.T @ (pred - labels) + reg @ (theta - prior)
        hess = (design.T * w) @ design + reg
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-7:
            break
    scale = float(max(0.05, theta[0]))
    bias = float(theta[1])
    return scale, bias


def select_threshold(
    probs: FloatArray, labels: FloatArray, *, max_fpr: float | None = None
) -> tuple[float, float, float]:
    """Pick the probability threshold maximizing F1, optionally constrained to FPR <= max_fpr.

    Returns (threshold, f1, fpr).
    """
    pos = labels > 0.5
    n_pos = int(np.sum(pos))
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5, 0.0, 0.0
    uniq = np.unique(probs)
    mids = 0.5 * (uniq[:-1] + uniq[1:]) if len(uniq) > 1 else np.array([], dtype=np.float64)
    candidates = np.concatenate(([0.5], uniq, mids))
    best_th = 0.5
    best_f1 = -1.0
    best_fpr = 1.0
    for th in cast("list[float]", candidates.tolist()):
        pred = probs >= th
        tp = int(np.sum(pred & pos))
        fp = int(np.sum(pred & ~pos))
        fn = n_pos - tp
        fpr = fp / n_neg
        if max_fpr is not None and fpr > max_fpr + 1e-12:
            continue
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and fpr < best_fpr):
            best_th = float(th)
            best_f1 = float(f1)
            best_fpr = float(fpr)
    if best_f1 < 0.0:
        best_th = float(np.max(probs)) + 1e-4
        best_f1 = 0.0
        best_fpr = 0.0
    return best_th, best_f1, best_fpr


def calibrate_scores(
    per_label: dict[str, tuple[list[float], list[int]]],
    *,
    max_fpr: float | None = None,
    platt: bool = True,
) -> list[LabelFit]:
    """Fit Platt calibration and operating thresholds for every label in `per_label`."""
    fits: list[LabelFit] = []
    for name in sorted(per_label):
        raw_scores, raw_labels = per_label[name]
        probs = np.asarray(raw_scores, dtype=np.float64)
        labels = np.asarray(raw_labels, dtype=np.float64)
        scale, bias = fit_platt(probs, labels) if platt else (1.0, 0.0)
        cal_probs = _sigmoid(scale * _logit(probs) + bias)
        th, f1, fpr = select_threshold(cal_probs, labels, max_fpr=max_fpr)
        fits.append(
            LabelFit(
                name=name,
                samples=len(raw_scores),
                positives=int(np.sum(labels > 0.5)),
                auroc=auroc(cal_probs, labels),
                scale=round(scale, 6),
                bias=round(bias, 6),
                threshold=round(float(np.clip(th, 1e-4, 1.0 - 1e-4)), 6),
                f1=f1,
                fpr=fpr,
            )
        )
    return fits


def _post_json(url: str, payload: dict[str, Any], timeout_s: float = 120.0) -> dict[str, Any]:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return cast("dict[str, Any]", json.loads(resp.read().decode()))


def _query_row(base_url: str, row: dict[str, Any], probe_model: str | None) -> dict[str, float]:
    """Score one JSONL row via /v1/moderations or /v1/chat/completions."""
    if "scores" in row and isinstance(row["scores"], dict):
        return {str(k): float(v) for k, v in cast("dict[str, Any]", row["scores"]).items()}
    base = base_url.rstrip("/")
    if "messages" in row:
        mod: dict[str, Any] = {}
        if probe_model:
            mod["model"] = probe_model
        payload: dict[str, Any] = {
            "messages": row["messages"],
            "max_tokens": int(row.get("max_tokens", 1)),
            "temperature": 0.0,
            "moderation": mod,
        }
        out = _post_json(f"{base}/v1/chat/completions", payload)
        scores: dict[str, float] = {}
        mod_out = cast("dict[str, Any]", out.get("moderation") or {})
        for side in ("input", "output"):
            side_obj = cast("dict[str, Any]", mod_out.get(side) or {})
            results = cast("list[dict[str, Any]]", side_obj.get("results") or [])
            if results:
                for k, v in cast("dict[str, Any]", results[0].get("category_scores") or {}).items():
                    scores[str(k)] = max(scores.get(str(k), 0.0), float(v))
        return scores
    payload = {"input": str(row["text"])}
    if probe_model:
        payload["model"] = probe_model
    out = _post_json(f"{base}/v1/moderations", payload)
    results = cast("list[dict[str, Any]]", out.get("results") or [])
    if not results:
        return {}
    return {str(k): float(v) for k, v in cast("dict[str, Any]", results[0].get("category_scores") or {}).items()}


def fit_logistic(
    features: FloatArray, labels: FloatArray, l2: float = 1.0, max_iter: int = 30
) -> tuple[npt.NDArray[np.float32], float]:
    """Fit L2-regularized logistic regression w*x + b via Newton-IRLS (dual form when D > N)."""
    n, d = features.shape
    y = labels.astype(np.float64)
    w = np.zeros(d, dtype=np.float64)
    b = float(np.log(max(float(np.sum(y > 0.5)), 0.5) / max(float(np.sum(y <= 0.5)), 0.5)))
    for _ in range(max_iter):
        z = features @ w + b
        p = _sigmoid(z)
        r = np.maximum(p * (1.0 - p), 1e-8)
        g_w = features.T @ (p - y) + l2 * w
        g_b = float(np.sum(p - y))
        if d <= n:
            # Primal Newton step on [w; b] of dimension d + 1.
            h = np.empty((d + 1, d + 1), dtype=np.float64)
            rx = features * r[:, None]
            h[:d, :d] = features.T @ rx + l2 * np.eye(d, dtype=np.float64)
            h[:d, d] = np.sum(rx, axis=0)
            h[d, :d] = h[:d, d]
            h[d, d] = float(np.sum(r)) + 1e-6
            g = np.append(g_w, g_b)
            step = np.linalg.solve(h, g)
            w -= step[:d]
            b -= float(step[d])
            diff = float(np.max(np.abs(step)))
        else:
            # Woodbury identity solves (X^T R X + l2 I) in O(N^2 D + N^3) when D >> N.
            k_mat = (features @ features.T) / l2
            sys_mat = np.diag(1.0 / r) + k_mat
            inv_h_g = (g_w - features.T @ np.linalg.solve(sys_mat, features @ g_w) / l2) / l2
            sum_rx = features.T @ r
            inv_h_u = (sum_rx - features.T @ np.linalg.solve(sys_mat, features @ sum_rx) / l2) / l2
            schur = float(np.sum(r)) + 1e-6 - float(sum_rx @ inv_h_u)
            step_b = (g_b - float(sum_rx @ inv_h_g)) / max(schur, 1e-8)
            step_w = inv_h_g - inv_h_u * step_b
            w -= step_w
            b -= step_b
            diff = max(float(np.max(np.abs(step_w))), abs(step_b))
        if diff < 1e-6:
            break
    return w.astype(np.float32), float(b)


def f32_to_bf16(arr: npt.NDArray[np.float32]) -> npt.NDArray[np.uint16]:
    """Round float32 values to bfloat16 (round-to-nearest-even)."""
    u32 = np.ascontiguousarray(arr, dtype="<f4").view("<u4")
    lsb = (u32 >> 16) & np.uint32(1)
    rounded = u32 + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype("<u2")


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    """Write a minimal safetensors file from {name: (dtype, shape, raw_bytes)}."""
    header: dict[str, Any] = {}
    offset = 0
    payloads: list[bytes] = []
    for name, (dtype, shape, raw) in tensors.items():
        end = offset + len(raw)
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, end]}
        payloads.append(raw)
        offset = end
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for raw in payloads:
            f.write(raw)


def write_linear_probe(
    out_dir: Path,
    weights: npt.NDArray[np.float32],
    biases: npt.NDArray[np.float32],
    *,
    model_name: str,
    hidden_size: int,
    layer_ids: list[int],
    label_specs: list[dict[str, Any]],
    rmsnorm_before_proj: bool = False,
) -> None:
    """Write a `c0ven_probe_v1` directory (`config.json` + `model.safetensors`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bf16_w = f32_to_bf16(weights)
    f32_b = np.ascontiguousarray(biases, dtype="<f4")
    cfg = {
        "model_type": "c0ven_probe_v1",
        "model_name": model_name,
        "hidden_size": hidden_size,
        "base_model_layer_ids": layer_ids,
        "rmsnorm_before_proj": rmsnorm_before_proj,
        "labels": label_specs,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    write_safetensors(
        out_dir / "model.safetensors",
        {
            "linear.weight": ("BF16", tuple(int(d) for d in weights.shape), bf16_w.tobytes()),
            "linear.bias": ("F32", tuple(int(d) for d in biases.shape), f32_b.tobytes()),
        },
    )


def _cmd_calibrate(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    per_label: dict[str, tuple[list[float], list[int]]] = {}
    for row in rows:
        scores = (
            _query_row(args.url, row, args.probe_model)
            if args.url
            else {str(k): float(v) for k, v in cast("dict[str, Any]", row.get("scores") or {}).items()}
        )
        labels = cast("dict[str, Any]", row.get("labels") or {})
        for name, target in labels.items():
            if name not in scores:
                continue
            bucket = per_label.setdefault(str(name), ([], []))
            bucket[0].append(float(scores[name]))
            bucket[1].append(int(bool(target)))
    if not per_label:
        raise SystemExit("no overlapping labels and scores found in dataset")
    fits = calibrate_scores(per_label, max_fpr=args.max_fpr, platt=not args.no_platt)
    cal: dict[str, Any] = {
        "roll_window": args.roll_window,
        "thresholds": {f.name: f.threshold for f in fits},
        "scale": {f.name: f.scale for f in fits},
        "bias": {f.name: f.bias for f in fits},
    }
    if args.probe_model:
        cal["model_name"] = args.probe_model
    if args.extra_alarms:
        cal["extra_alarms"] = args.extra_alarms
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cal, indent=2) + "\n")
    for f in fits:
        print(
            f"{f.name:28s} n={f.samples:4d} pos={f.positives:4d} auroc={f.auroc:.4f} "
            f"scale={f.scale:.3f} bias={f.bias:+.3f} th={f.threshold:.4f} f1={f.f1:.3f} fpr={f.fpr:.3f}"
        )
    return 0


def _cmd_fit_linear(args: argparse.Namespace) -> int:
    data = np.load(args.data)
    if args.scope == "drift":
        if "X_before" not in data or "X_after" not in data:
            raise SystemExit("drift mode requires X_before and X_after arrays in --data")
        x = np.asarray(data["X_after"], dtype=np.float64) - np.asarray(data["X_before"], dtype=np.float64)
    else:
        if "X" not in data:
            raise SystemExit("--data must contain X and y arrays")
        x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.float64)
    if y.ndim == 1:
        y = y[:, None]
    names = [s.strip() for s in args.labels.split(",") if s.strip()]
    if len(names) != y.shape[1]:
        raise SystemExit(f"got {len(names)} label name(s) for y with {y.shape[1]} column(s)")
    layer_ids = [int(s) for s in args.layers.split(",") if s.strip()]
    if x.shape[1] != len(layer_ids) * args.hidden_size:
        raise SystemExit(f"feature dim {x.shape[1]} != len(layers)*hidden_size ({len(layer_ids) * args.hidden_size})")
    ws: list[npt.NDArray[np.float32]] = []
    bs: list[float] = []
    specs: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        w, b = fit_logistic(x, y[:, i], l2=args.l2)
        probs = _sigmoid(x @ w.astype(np.float64) + b)
        th, f1, fpr = select_threshold(probs, y[:, i], max_fpr=args.max_fpr)
        ws.append(w)
        bs.append(b)
        specs.append({"name": name, "scope": args.scope, "threshold": round(th, 6)})
        print(f"{name:28s} auroc={auroc(probs, y[:, i]):.4f} th={th:.4f} f1={f1:.3f} fpr={fpr:.3f}")
    write_linear_probe(
        args.out_dir,
        np.stack(ws, axis=0),
        np.asarray(bs, dtype=np.float32),
        model_name=args.model_name or args.out_dir.name,
        hidden_size=args.hidden_size,
        layer_ids=layer_ids,
        label_specs=specs,
        rmsnorm_before_proj=args.rmsnorm,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.probe_calibrate", description=(__doc__ or "").splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calibrate", help="fit Platt parameters and thresholds and write calibration.json")
    c.add_argument("dataset", type=Path, help="JSONL with text/messages, labels, and optional precomputed scores")
    c.add_argument("--out", type=Path, required=True, help="output calibration.json path")
    c.add_argument("--url", default="", help="running c0ven server base URL (omit if dataset already holds scores)")
    c.add_argument("--probe-model", default=None, help="override probe model_name in calibration.json")
    c.add_argument("--roll-window", type=int, default=10, help="rolling-mean token window (default: 10)")
    c.add_argument("--max-fpr", type=float, default=None, help="optional maximum false positive rate for thresholds")
    c.add_argument("--no-platt", action="store_true", help="keep scale=1, bias=0 and fit thresholds only")
    c.add_argument("--extra-alarms", default=None, help="optional c0ven_probe_v1 sidecar directory path")

    f = sub.add_parser("fit-linear", help="fit a c0ven_probe_v1 linear or drift probe from .npz hidden states")
    f.add_argument("data", type=Path, help=".npz holding X,y (or X_before,X_after,y for --scope drift)")
    f.add_argument("--out-dir", type=Path, required=True, help="output probe directory")
    f.add_argument("--labels", required=True, help="comma-separated label names matching y columns")
    f.add_argument("--layers", default="15,31,47", help="comma-separated tapped 0-based layer ids")
    f.add_argument("--hidden-size", type=int, default=2560, help="model hidden dimension per tapped layer")
    f.add_argument("--scope", choices=("input", "output", "both", "drift"), default="input")
    f.add_argument("--l2", type=float, default=1.0, help="L2 regularization strength")
    f.add_argument("--max-fpr", type=float, default=None, help="optional max FPR for fitted threshold")
    f.add_argument("--model-name", default=None, help="probe model name in config.json")
    f.add_argument("--rmsnorm", action="store_true", help="set rmsnorm_before_proj=true in config.json")

    args = p.parse_args(argv)
    if args.cmd == "calibrate":
        return _cmd_calibrate(args)
    return _cmd_fit_linear(args)


if __name__ == "__main__":
    raise SystemExit(main())
