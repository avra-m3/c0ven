"""The tokenizer and the incremental detokenizer (ADR-0004: Hugging Face `tokenizers`).

The model ships a Hugging Face `tokenizer.json`. The GGUF carries the same vocabulary and merges
(`tokenizer.ggml.*`), so without the JSON the tokenizer is rebuilt from them, as the JSON describes it: NFC, the
pre-tokenizer regex of spec/MODEL_IO.md, section 4.1, byte-level BPE, and the added tokens matched whole. Either way,
reference/prompts.jsonl checks it end to end (tests/end_to_end/qwen4exp/test_tokenizer.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

PRETOKENIZE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+"
    r"|\s+(?!\S)|\s+"
)
"""tokenizer_config.json's `pretokenize_regex`."""

CONTROL = 3
USER_DEFINED = 4
"""llama.cpp's token types for special and other added tokens (`tokenizer.ggml.token_type`)."""


def from_json(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def from_vocabulary(tokens: Sequence[str], merges: Sequence[str], token_types: Sequence[int]) -> Tokenizer:
    """The tokenizer the GGUF's `tokenizer.ggml.tokens`, `merges` and `token_type` describe."""
    added = {i for i, t in enumerate(token_types) if t in (CONTROL, USER_DEFINED)}
    vocab = {tok: i for i, tok in enumerate(tokens)}
    pairs: list[tuple[str, str]] = []
    for m in merges:
        a, sep, b = m.partition(" ")
        if not sep:
            raise ValueError(f"bad merge: {m!r}")
        pairs.append((a, b))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=pairs, fuse_unk=False, byte_fallback=False))
    tk.normalizer = normalizers.NFC()
    tk.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(PRETOKENIZE), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tk.decoder = decoders.ByteLevel()
    specials = [AddedToken(tokens[i], special=token_types[i] == CONTROL, normalized=False) for i in sorted(added)]
    tk.add_tokens(specials)  # pyright: ignore[reportUnknownMemberType]: the stub's list is untyped
    for i in sorted(added):
        if tk.token_to_id(tokens[i]) != i:
            raise ValueError(f"added token {tokens[i]!r} landed at {tk.token_to_id(tokens[i])}, not {i}")
    return tk


def encode(tk: Tokenizer, text: str) -> list[int]:
    """Special tokens in the rendered text are matched whole; nothing is added (the model has no BOS)."""
    return list(tk.encode(text, add_special_tokens=False).ids)


@dataclass(slots=True)
class Detokenizer:
    """Turns token ids into text as they arrive, holding back a token that ends inside a UTF-8 character.

    Decodes a short window behind the text already sent and emits only what follows it, the method of HF's
    `TextStreamer` and vLLM's incremental detokenizer. Special tokens are kept, so the output parser sees
    `<think>` and `<tool_call>`."""

    tokenizer: Tokenizer
    ids: list[int] = field(default_factory=lambda: [])
    _prefix: int = 0
    _read: int = 0

    def push(self, token_ids: Sequence[int]) -> str:
        self.ids.extend(token_ids)
        prefix_text = self._decode(self.ids[self._prefix : self._read])
        text = self._decode(self.ids[self._prefix :])
        if text.endswith("�") or len(text) <= len(prefix_text):
            return ""
        new = text[len(prefix_text) :]
        self._prefix = self._read
        self._read = len(self.ids)
        return new

    def flush(self) -> str:
        """Whatever is held back, replacement characters and all: the end of the output."""
        prefix_text = self._decode(self.ids[self._prefix : self._read])
        text = self._decode(self.ids[self._prefix :])
        self._prefix = self._read = len(self.ids)
        return text[len(prefix_text) :]

    def _decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)
