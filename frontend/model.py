"""What the front end reads from the model's files: the tokenizer, the chat template, the stop tokens, the defaults.

From the GGUF's metadata (MODEL_FILES.md, "Metadata keys"), unless a `tokenizer.json` or a `chat_template.jinja` is
given. The front end reads the file's header only; the tensors are the engine's.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import gguf
from tokenizers import Tokenizer

import protocol
from frontend.template import ChatTemplate
from frontend.tokenizer import from_json, from_vocabulary


@dataclass(frozen=True, slots=True)
class Model:
    name: str
    tokenizer: Tokenizer
    template: ChatTemplate
    stop_ids: frozenset[int]
    """Tokens that end the output and are not shown: the end of a turn and the end of text."""
    defaults: protocol.Settings
    path: str = ""
    arch: str = ""
    """The GGUF's `general.architecture`, passed to the engine so it need not read the header again."""


def _field(reader: gguf.GGUFReader, key: str) -> object | None:
    f = reader.get_field(key)
    return None if f is None else f.contents()


def _array(reader: gguf.GGUFReader, key: str) -> list[str | int]:
    value = _field(reader, key)
    if not isinstance(value, list):
        raise ValueError(f"the GGUF has no {key}")
    return cast("list[str | int]", value)


def _stop_ids(tokenizer: Tokenizer, *ids: object) -> frozenset[int]:
    out = {i for i in ids if isinstance(i, int)}
    for token in ("<|im_end|>", "<|endoftext|>"):
        if (i := tokenizer.token_to_id(token)) is not None:
            out.add(i)
    return frozenset(out)


def load(
    gguf_path: Path | None,
    tokenizer_json: Path | None = None,
    chat_template: Path | None = None,
    name: str | None = None,
) -> Model:
    if gguf_path is None:
        if tokenizer_json is None or chat_template is None:
            raise ValueError("without a GGUF file, give both tokenizer.json and the chat template")
        tokenizer = from_json(tokenizer_json)
        return Model(
            name or tokenizer_json.parent.name,
            tokenizer,
            ChatTemplate(chat_template.read_text()),
            _stop_ids(tokenizer),
            protocol.Settings(),
        )
    reader = gguf.GGUFReader(gguf_path)
    if tokenizer_json is not None:
        tokenizer = from_json(tokenizer_json)
    else:
        tokens = _array(reader, "tokenizer.ggml.tokens")
        merges = _array(reader, "tokenizer.ggml.merges")
        types = _array(reader, "tokenizer.ggml.token_type")
        tokenizer = from_vocabulary([str(t) for t in tokens], [str(m) for m in merges], [int(t) for t in types])
    source = chat_template.read_text() if chat_template else _field(reader, "tokenizer.chat_template")
    if not isinstance(source, str):
        raise ValueError(f"{gguf_path} has no chat template")
    defaults = protocol.Settings()
    temp, top_k, top_p = (_field(reader, f"general.sampling.{k}") for k in ("temp", "top_k", "top_p"))
    defaults = protocol.Settings(
        temperature=float(temp) if isinstance(temp, (int, float)) else defaults.temperature,
        top_k=int(top_k) if isinstance(top_k, int) else defaults.top_k,
        top_p=float(top_p) if isinstance(top_p, (int, float)) else defaults.top_p,
    )
    model_name = name or str(_field(reader, "general.name") or gguf_path.stem)
    return Model(
        model_name,
        tokenizer,
        ChatTemplate(source),
        _stop_ids(
            tokenizer,
            _field(reader, "tokenizer.ggml.eos_token_id"),
            _field(reader, "tokenizer.ggml.padding_token_id"),
        ),
        defaults,
        str(gguf_path),
        str(_field(reader, "general.architecture") or ""),
    )
