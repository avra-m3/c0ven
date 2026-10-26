"""qwen4exp's products and measured terms for the step-time model (engine/step_time.py).

The products come from the tensor table, their bytes from each tensor's shape and its quant type's block size; how
each splits is split.py's rule. The terms are the measured ones, with the load check's timings where there is one,
calibrated on the committed IQ3_S table.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from functools import cache

import gguf

from engine.api.interface import LoadCheck, SplitKind
from engine.models.qwen4exp import numbers
from engine.models.qwen4exp.split import split_of
from engine.models.qwen4exp.tensors import FileTensor, LayerKind, TensorTable, from_tsv
from engine.step_time import Class, ModelSteps, Product, Rates, Terms, calibrate

_GATE_UP_AS = {"IQ4_XS": "IQ3_S"}
"""Expert types with no measured rate, and the measured type each is costed as."""

_DENSE = frozenset(
    {
        "attn_qkv",
        "attn_q",
        "attn_k",
        "attn_v",
        "attn_gate",
        "attn_output",
        "ssm_out",
        "ffn_gate_shexp",
        "ffn_up_shexp",
        "ffn_down_shexp",
    }
)
"""The dense products a step's time counts. The router, the norms, the hyper-connections and the indexer are in F."""

_MEASURED_DENSE = {
    "attn_qkv": ("Q6_K", (2560, 10240)),
    "ssm_out": ("Q6_K", (6144, 2560)),
    "attn_output": ("Q4_K", (6144, 2560)),
    "attn_gate": ("Q4_K", (2560, 6144)),
    "ffn_gate_shexp": ("Q4_K", (2560, 640)),
    "ffn_down_shexp": ("IQ4_NL", (640, 2560)),
}
"""The type and shape, in ggml's order, of the tensor each dense rate was measured on."""

_BLK = re.compile(r"^blk\.\d+\.(?P<name>.+)\.weight$")


def _block(t: FileTensor) -> tuple[int, int]:
    """(values, bytes) of one quant block of the tensor's type."""
    return gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(t.ggml_type)]


def _nbytes(t: FileTensor) -> int:
    values, size = _block(t)
    n = math.prod(t.shape) // values * size
    if n != t.nbytes:
        raise ValueError(f"{t.name}: {t.nbytes:,} B in the table, {n:,} from its shape and type")
    return n


def _bits(type_name: str) -> float:
    values, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[type_name]]
    return 8 * size / values


def _dense_kernel(t: FileTensor) -> str:
    """The nearest measured product: of those split the same way, the nearest in size, then in bits per weight.
    Small products take a small product's rate, since launches, not bytes, bound them."""
    kind = split_of(t.name)
    same = [p for p in _MEASURED_DENSE if split_of(f"blk.0.{p}.weight") is kind]
    size = math.prod(t.shape)
    gap = {p: abs(math.log(math.prod(_MEASURED_DENSE[p][1]) / size)) for p in same}
    nearest = [p for p in same if math.isclose(gap[p], min(gap.values()))]
    return min(nearest, key=lambda p: abs(_bits(_MEASURED_DENSE[p][0]) - _bits(t.type_name)))


def model_steps(table: TensorTable) -> ModelSteps:
    """The products of each layer and the head. Raises ValueError for a table whose bytes do not match its shapes
    or whose gate and up experts differ in type."""
    names = table.by_name()
    per_token = numbers.EXPERTS_A_TOKEN
    experts = numbers.EXPERTS_PER_LAYER
    layers: list[tuple[Product, ...]] = []
    for n in range(numbers.LAYERS):
        gate, up, down = (names[f"blk.{n}.ffn_{p}_exps.weight"] for p in ("gate", "up", "down"))
        if gate.type_name != up.type_name:
            raise ValueError(f"layer {n}: gate {gate.type_name}, up {up.type_name}")
        gu = _GATE_UP_AS.get(gate.type_name, gate.type_name)
        products = [
            Product(f"gate_up {gu}", Class.GATE_UP, per_token * (_nbytes(gate) + _nbytes(up)) // experts, True, 0, 1),
            Product(
                f"down {down.type_name}",
                Class.DOWN,
                per_token * _nbytes(down) // experts,
                True,
                down.shape[0],
                _block(down)[0],
            ),
        ]
        for t in table.tensors:
            m = _BLK.match(t.name)
            if t.layer == n and m and m.group("name") in _DENSE:
                kernel = _dense_kernel(t)
                cols = t.shape[0] if split_of(t.name) is SplitKind.COLUMNS else 0
                products.append(Product(kernel, Class.DENSE, _nbytes(t), False, cols, _block(t)[0]))
        layers.append(tuple(products))
    head = names["output.weight"]
    attention = sum(k is LayerKind.ATTENTION for k in table.layer_kinds())
    return ModelSteps(
        layers=tuple(layers),
        head=(Product("output", Class.HEAD, _nbytes(head), False, 0, _block(head)[0]),),
        # Two reduces a layer, and a third for the indexer in each attention layer (DESIGN.md, section 5).
        exchanges=2 * numbers.LAYERS + attention,
        experts=experts,
        experts_a_token=per_token,
        kv_heads=numbers.KV_HEADS,
        list_rows=(1, 2, 4, 8),  # Model::record's widths
    )


# Each value's row in docs/measurements/MEASUREMENTS.md: the rates #2026-10-25-slice-width-kernels; the hit-rate
# curve #2026-10-25-hit-rate-against-experts-cached; the miss cost #2026-10-16-cpu-answer-per-layer; the peer
# writes the middle of #2026-10-16-peer-exchange's ranges; the steps, the CPU's wait and F's growth a row
# #2026-10-19-four-rows-a-step. The hand-off is an estimate from #2026-10-12-card-to-card.
MEASURED = Terms(
    widths=(1, 2, 4, 8),
    rates={
        "gate_up IQ3_S": {1: (255, 253, 180, 101), 2: (260, 225, 229, 162), 4: (294, 275, 240, 200)},
        "gate_up IQ3_XXS": {1: (261, 247, 175, 96), 2: (266, 239, 239, 168), 4: (300, 283, 251, 208)},
        "gate_up IQ2_S": {1: (198, 207, 153, 92), 2: (215, 188, 195, 145), 4: (252, 233, 202, 183)},
        "down IQ4_NL": {1: (277, 217, 164, None), 2: (291, 237, 168, None), 4: (293, 233, 167, None)},
        "down Q2_0": {1: (259, 228, None, None), 2: (274, 209, None, None), 4: (224, 182, None, None)},
        "attn_qkv": {1: (408, 387, 351, 293)},
        "ssm_out": {1: (354, 354, 307, 228)},
        "attn_output": {1: (259, 230, 182, 142)},
        "attn_gate": {1: (221, 171, 186, 113)},
        "ffn_gate_shexp": {1: (96, 82, 49, 25)},
        "ffn_down_shexp": {1: (148, 101, 71, None)},
        "output": {1: (446, 444, 440, 429)},
    },
    reference="gate_up IQ3_S",
    exchange_us={1: 2.9, 4: 5.1},
    handoff_us=17.5,
    hit_rate=(
        (78, 0.496),
        (128, 0.648),
        (200, 0.798),
        (248, 0.867),
        (307, 0.926),
        (317, 0.934),
        (369, 0.966),
        (400, 0.979),
        (460, 0.995),
        (481, 0.998),
        (512, 1.0),
    ),
    miss_layer_us=22.5,
    miss_expert_us=61.0,
    cpu_wait_ms={1: 0.24, 4: 5.70},
    cpu_wait_experts=317,
    step_ms=15.50,
    fixed_per_row_ms=0.61,
)
"""The terms as measured, before calibration."""


@cache
def measured_terms() -> Terms:
    """MEASURED with F(1) calibrated on the IQ3_S file, the file the step was measured on."""
    return calibrate(MEASURED, model_steps(from_tsv()))


def _group(kernel: str) -> str:
    """The load check's kernel a rate follows: the experts' gate and up, their down, or matvec for the rest."""
    return kernel.split(" ", maxsplit=1)[0] if kernel.startswith(("gate_up ", "down ")) else "dense"


def with_check(terms: Terms, check: LoadCheck) -> Terms:
    """The terms with the load check's timings. At each width and row count the check timed, every rate of a group
    (gate and up, down, the dense products and the head) is scaled by how the group's timed kernel compares with its
    own rate here; a row count it did not time takes the one-row change. Widths it did not time keep their rates. A
    timed kernel with no rate here enters as timed, and scales nothing else. The check's peer writes and hand-off
    replace these."""
    timed: dict[str, tuple[str, dict[int, dict[int, float]]]] = {
        "gate_up": (f"gate_up {check.gate_up_type}", check.gate_up_gbs),
        "down": (f"down {check.down_type}", check.down_gbs),
        "dense": ("attn_qkv", {w: {1: v} for w, v in check.dense_gbs.items()}),
    }
    n = len(terms.widths)
    rates = {k: {r: list(v) for r, v in by.items()} for k, by in terms.rates.items()}
    for group, (key, measured) in timed.items():
        own = terms.rates.get(key)
        if own is None:
            rows = sorted({r for by in measured.values() for r in by})
            rates[key] = {r: [measured.get(w, {}).get(r) for w in terms.widths] for r in rows}
            continue
        change: dict[tuple[int, int], float] = {}
        for w, by in measured.items():
            if w not in terms.widths:
                continue
            i = terms.widths.index(w)
            for r, gbs in by.items():
                was = own.get(r, (None,) * n)[i]
                if was:
                    change[i, r] = gbs / was
        for k, by in rates.items():
            if _group(k) != group:
                continue
            for r, values in by.items():
                for i, v in enumerate(values):
                    f = change.get((i, r), change.get((i, 1)))
                    if v is not None and f is not None:
                        values[i] = v * f
    out: dict[str, Rates] = {k: {r: tuple(v) for r, v in by.items()} for k, by in rates.items()}
    return replace(
        terms,
        rates=out,
        exchange_us=check.exchange_us or terms.exchange_us,
        handoff_us=terms.handoff_us if check.handoff_us is None else check.handoff_us,
        fixed_ms=None,
    )


def checked_terms(check: LoadCheck) -> Terms:
    """MEASURED with the check's timings, F(1) calibrated as `measured_terms` calibrates it."""
    return calibrate(with_check(MEASURED, check), model_steps(from_tsv()))
