"""The layout's import rules (ADR-0024, items 3 and 7)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.add(node.module)
    return out


def test_the_front_end_never_imports_the_engine():
    offenders = {
        str(p.relative_to(ROOT)): sorted(m for m in imported_modules(p) if m == "engine" or m.startswith("engine."))
        for p in (ROOT / "frontend").rglob("*.py")
    }
    assert {k: v for k, v in offenders.items() if v} == {}


def test_the_engine_outside_models_does_not_name_the_model():
    offenders = [
        str(p.relative_to(ROOT))
        for p in (ROOT / "engine").rglob("*.py")
        if "models" not in p.relative_to(ROOT / "engine").parts and "qwen4exp" in p.read_text()
    ]
    assert offenders == []


def test_protocol_imports_nothing_of_ours():
    ours = {"engine", "frontend", "tools", "tests"}
    assert {m.split(".")[0] for m in imported_modules(ROOT / "protocol.py")} & ours == set()
