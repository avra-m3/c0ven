"""Build reference prompt rows from chat messages, with the model's own template and tokenizer (REFERENCE_OUTPUTS.md).

    uv run tools/reference_prompts.py --model <first shard>.gguf reference/messages.jsonl reference/prompts.jsonl
    uv run tools/reference_prompts.py --model <first shard>.gguf --check reference/prompts.jsonl

Each line of the messages file is one prompt: `id`, `kind`, `thinking`, `effort`, `max_tokens`, `messages` and `tools`
in the shape `/v1/chat/completions` takes. The row written for it carries the rendered `text` and its `tokens`. A row
whose id the output file already has is replaced; the others stay. `--check` renders nothing: it tokenizes every
row's text again and reports the rows whose tokens differ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontend.model import Model, load
from frontend.template import Conversation
from frontend.tokenizer import encode

type Row = dict[str, Any]

KEYS = ("id", "kind", "thinking", "effort", "max_tokens")


def read_rows(path: Path) -> list[Row]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path: Path, rows: list[Row]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build(model: Model, spec: Row) -> Row:
    """The prompt row for one messages row: the template's text and the tokenizer's ids."""
    conversation = Conversation(
        spec["messages"],
        tools=spec.get("tools"),
        enable_thinking=None if spec["thinking"] else False,
        reasoning_effort=spec.get("effort"),
    )
    text = model.template.render(conversation)
    tokens = encode(model.tokenizer, text)
    row: Row = {k: spec.get(k) for k in KEYS}
    row.update(text=text, n_tokens=len(tokens), tokens=tokens)
    return row


def check(model: Model, rows: list[Row]) -> list[str]:
    """The ids whose text does not tokenize to their tokens."""
    return [row["id"] for row in rows if encode(model.tokenizer, row["text"]) != row["tokens"]]


def merge(existing: list[Row], built: list[Row]) -> list[Row]:
    """`existing` with each built row in place of the row of the same id, new ids at the end."""
    by_id = {row["id"]: row for row in built}
    out = [by_id.pop(row["id"], row) for row in existing]
    return out + [by_id[k] for k in by_id]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, help="a GGUF shard; its vocabulary and chat template are used")
    p.add_argument("--tokenizer-json", type=Path, help="a tokenizer.json, in place of the GGUF's vocabulary")
    p.add_argument("--chat-template", type=Path, help="a chat_template.jinja, in place of the GGUF's template")
    p.add_argument("--check", action="store_true", help="tokenize the prompt file's rows again and report differences")
    p.add_argument("messages", type=Path, help="the messages file (or, with --check, the prompt file)")
    p.add_argument("prompts", type=Path, nargs="?", help="the prompt file to write or update")
    args = p.parse_args()
    model = load(args.model, args.tokenizer_json, args.chat_template)
    if args.check:
        bad = check(model, read_rows(args.messages))
        print("every row's text tokenizes to its tokens" if not bad else "tokens differ: " + ", ".join(bad))
        return 1 if bad else 0
    if args.prompts is None:
        p.error("give the prompt file to write")
    built = [build(model, spec) for spec in read_rows(args.messages)]
    existing = read_rows(args.prompts) if args.prompts.exists() else []
    write_rows(args.prompts, merge(existing, built))
    for row in built:
        print(f"{row['id']:16s} {row['n_tokens']:6d} tokens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
