"""
Pluggable tokenizer for the conversation-budget progress bar in the UI.

`build_tokenizer(settings)` returns a `Tokenizer` whose `count(text)` returns
an integer token estimate. Three backends, picked via TOKENIZER_METHOD:

  - transformers : `AutoTokenizer.from_pretrained(...)` — exact for the
                   target model, but `transformers` is a heavy install. Lazy
                   import so this module doesn't fail on import when the
                   package isn't there.
  - tiktoken     : Encoded with `cl100k_base` by default. Close-enough for a
                   progress bar; ships with langchain-openai already.
  - char_count   : `len(text) // 2`. Zero deps. Used as the universal
                   fallback when whichever backend was requested can't load.

The fallback to char_count happens at startup (not per request), so a missing
`transformers` install doesn't make every chat turn raise.
"""

from __future__ import annotations

import logging
from typing import Iterable, Protocol

from app.config import Settings, TokenizerMethod

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    method: str

    def count(self, text: str) -> int: ...

    def count_many(self, texts: Iterable[str]) -> int: ...


class _CharCountTokenizer:
    method = "char_count"

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(text) // 2

    def count_many(self, texts: Iterable[str]) -> int:
        return sum(self.count(t) for t in texts)


class _TiktokenTokenizer:
    method = "tiktoken"

    def __init__(self) -> None:
        import tiktoken  # local import keeps the module importable without it
        self._enc = tiktoken.get_encoding("cl100k_base")

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def count_many(self, texts: Iterable[str]) -> int:
        return sum(self.count(t) for t in texts)


class _TransformersTokenizer:
    method = "transformers"

    def __init__(self, model_name: str) -> None:
        from transformers import AutoTokenizer  # heavy; lazy
        self._tok = AutoTokenizer.from_pretrained(model_name)
        self._model_name = model_name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_many(self, texts: Iterable[str]) -> int:
        return sum(self.count(t) for t in texts)


def build_tokenizer(settings: Settings) -> Tokenizer:
    method = settings.tokenizer_method
    try:
        if method == TokenizerMethod.TRANSFORMERS:
            tok = _TransformersTokenizer(settings.tokenizer_model_name)
            logger.info(
                "Tokenizer: transformers (%s) loaded",
                settings.tokenizer_model_name,
            )
            return tok
        if method == TokenizerMethod.TIKTOKEN:
            tok = _TiktokenTokenizer()
            logger.info("Tokenizer: tiktoken (cl100k_base)")
            return tok
    except Exception as e:
        logger.warning(
            "Tokenizer init failed for method=%s (%s); falling back to char_count",
            method.value, e,
        )

    logger.info("Tokenizer: char_count (fallback or selected)")
    return _CharCountTokenizer()
