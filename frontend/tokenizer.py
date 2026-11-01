"""The tokenizer and the incremental detokenizer (ADR-0004: Hugging Face `tokenizers`).

The model ships a Hugging Face `tokenizer.json`. The GGUF carries the same vocabulary and merges
(`tokenizer.ggml.*`), so without the JSON the tokenizer is rebuilt from them, as the JSON describes it: the normalizer
and pre-tokenizer that `tokenizer.ggml.pre` names, byte-level BPE, and the added tokens matched whole. For Flash-Next,
reference/prompts.jsonl checks it end to end (tests/end_to_end/qwen4exp/test_model_text.py).
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
"""Flash-Next's (`qwen35`): tokenizer_config.json's `pretokenize_regex`."""

JOYAI_LLM = (
    r"\p{N}{1,3}",
    "[\u4e00-\u9fa5\u3040-\u309f\u30a0-\u30ff]+",
    r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*"
    r"|\s*[\r\n]+|\s+(?!\S)|\s+",
)
"""DeepSeek-V3's splits (`joyai-llm`), applied in turn: runs of up to three digits, CJK runs, then words and
punctuation. DeepSeek-V4.1-Flash's tokenizer.json has these and no normalizer; llama.cpp gives `joyai-llm` the same
(`src/llama-vocab.cpp:319-328` at 3cf0325)."""

PRE_TOKENIZERS = ("qwen35", "joyai-llm")
"""The `tokenizer.ggml.pre` values from_vocabulary builds."""

CONTROL = 3
USER_DEFINED = 4
"""llama.cpp's token types for special and other added tokens (`tokenizer.ggml.token_type`)."""


def from_json(path: Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))


def from_vocabulary(tokens: Sequence[str], merges: Sequence[str], token_types: Sequence[int], pre: str) -> Tokenizer:
    """The tokenizer the GGUF's `tokenizer.ggml.tokens`, `merges`, `token_type` and `pre` describe.

    Raises ValueError for a `pre` not in PRE_TOKENIZERS, a malformed merge, or an added token whose id is not its
    index in `tokens`."""
    if pre not in PRE_TOKENIZERS:
        raise ValueError(
            f"tokenizer.ggml.pre {pre!r} is not one the front end builds ({', '.join(PRE_TOKENIZERS)}); "
            "give the model's tokenizer.json with --tokenizer"
        )
    added = {i for i, t in enumerate(token_types) if t in (CONTROL, USER_DEFINED)}
    vocab = {tok: i for i, tok in enumerate(tokens)}
    pairs: list[tuple[str, str]] = []
    for m in merges:
        a, sep, b = m.partition(" ")
        if not sep:
            raise ValueError(f"bad merge: {m!r}")
        pairs.append((a, b))
    tk = Tokenizer(models.BPE(vocab=vocab, merges=pairs, fuse_unk=False, byte_fallback=False))
    if pre == "qwen35":
        tk.normalizer = normalizers.NFC()
        splits = [PRETOKENIZE]
    else:
        splits = list(JOYAI_LLM)
    tk.pre_tokenizer = pre_tokenizers.Sequence(
        [
            *(pre_tokenizers.Split(Regex(r), behavior="isolated", invert=False) for r in splits),
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
