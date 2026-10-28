"""Build the KL set, reference/kl-set.jsonl: five segments of 1,000 tokens read teacher-forced (REFERENCE_OUTPUTS.md).

    uv run tools/kl_set.py --tokenizer tokenizer.json reference/kl-set.jsonl

The code and docs segments are the first 1,000 tokens of a file of this repository; English prose and the four
languages are reference/kl-text/; the agent segment is the 1,000 tokens of `agent-turn4` from the message start
whose window holds the most turns. Each row holds the segment's `id`, its `source`, its `tokens` and their decoded
`text`. The rows hold the tokens the engine and the base read, so the files they come from may change afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.tokenizer import Tokenizer, encode, from_json

ROOT = Path(__file__).resolve().parents[1]
N = 1000
LANGUAGES = ("german", "french", "japanese", "chinese")
IM_START = "<|im_start|>"


def head(tk: Tokenizer, path: Path, n: int) -> list[int]:
    tokens = encode(tk, path.read_text())
    if len(tokens) < n:
        raise ValueError(f"{path}: {len(tokens)} tokens, fewer than {n}")
    return tokens[:n]


def agent_window(tk: Tokenizer, prompts: Path, n: int) -> tuple[list[int], int]:
    """The n tokens of agent-turn4 from the message start whose window holds the most message starts (the first such
    start), and that start's index."""
    row = next(r for r in map(json.loads, prompts.read_text().splitlines()) if r["id"] == "agent-turn4")
    tokens: list[int] = row["tokens"]
    im = tk.token_to_id(IM_START)
    starts = [i for i, t in enumerate(tokens) if t == im and len(tokens) - i >= n]
    first = max(starts, key=lambda s: (sum(1 for i in range(s, s + n) if tokens[i] == im), -s))
    return tokens[first : first + n], first


def build(tk: Tokenizer) -> list[dict[str, Any]]:
    text_dir = ROOT / "reference" / "kl-text"
    agent, first = agent_window(tk, ROOT / "reference" / "prompts.jsonl", N)
    segments = [
        ("code", "core/models/qwen4exp/swap.cpp", head(tk, ROOT / "core/models/qwen4exp/swap.cpp", N)),
        ("docs", "docs/CACHING.md", head(tk, ROOT / "docs/CACHING.md", N)),
        ("english", "reference/kl-text/english.txt", head(tk, text_dir / "english.txt", N)),
        (
            "languages",
            ", ".join(f"reference/kl-text/{lang}.txt" for lang in LANGUAGES),
            [t for lang in LANGUAGES for t in head(tk, text_dir / f"{lang}.txt", N // len(LANGUAGES))],
        ),
        ("agent", f"reference/prompts.jsonl, agent-turn4, tokens {first} to {first + N - 1}", agent),
    ]
    return [
        {
            "id": sid,
            "source": src,
            "n_tokens": len(toks),
            "tokens": toks,
            "text": tk.decode(toks, skip_special_tokens=False),
        }
        for sid, src, toks in segments
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", type=Path, required=True, help="the model's tokenizer.json")
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    rows = build(from_json(args.tokenizer))
    with args.out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")  # kl_test reads it
    for row in rows:
        print(f"{row['id']:<10} {row['n_tokens']} tokens  {row['source']}")


if __name__ == "__main__":
    main()
