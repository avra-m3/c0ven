"""Print the memory plan for a model and a set of options (DESIGN.md, section 6), without starting anything.

    python -m tools.plan --slots 4 --context 262144
    python -m tools.plan --model /models/<file>-00001-of-00002.gguf --slots 1 --context 32768

Without `--model` it uses the committed IQ3_S tensor table (docs/model/tensors-IQ3_S.tsv).
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import cast

import protocol
from engine.memory_plan import MemoryPlan, PlanError, PlanOptions


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tools.plan", description=(__doc__ or "").splitlines()[0])
    p.add_argument("--model", type=Path)
    p.add_argument("--arch", default="qwen4exp")
    p.add_argument("--slots", type=int, default=1)
    p.add_argument("--context", type=int, default=262_144)
    p.add_argument("--checkpoints", type=int, default=8)
    p.add_argument(
        "--kv-tokens", type=protocol.tokens, default=0, help="the KV pool the slots share; default: slots x context"
    )
    args = p.parse_args(argv)
    setup = importlib.import_module(f"engine.models.{args.arch}.setup")
    options = PlanOptions(
        slots=args.slots, context=args.context, checkpoints=args.checkpoints, kv_tokens=args.kv_tokens
    )
    try:
        _, plan = cast("tuple[object, MemoryPlan]", setup.open_config(args.model, options))
    except PlanError as exc:
        print(f"does not fit: {exc}")
        return 1
    width = max(len(name) for name, _ in plan.rows())
    for name, value in plan.rows():
        print(f"{name:<{width}}  {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
