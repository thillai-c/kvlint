"""Tokenizer loading and token/offset extraction.

Everything downstream depends on the token ids here matching what the engine
computes for the same prompt. M6 checks that claim against a live server's
``usage.prompt_tokens``.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from kvlint.errors import TokenizerError


@lru_cache(maxsize=8)
def load_tokenizer(model: str) -> Any:
    """Load a fast HuggingFace tokenizer, cached per model id.

    Loading is the slowest step in a run, and `analyze` tokenizes thousands of
    requests with one model, so the cache matters more than it looks.
    """
    # transformers prints a banner about missing PyTorch on import. We only ever
    # use tokenizers and chat templates, so the banner is noise in CLI output.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise TokenizerError("transformers is required to tokenize requests") from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model)
    except Exception as exc:
        raise TokenizerError(f"could not load tokenizer for {model!r}: {exc}") from exc

    if not getattr(tokenizer, "is_fast", False):
        # Slow tokenizers cannot return offset mappings, and without offsets a
        # lint finding cannot quote the text around a divergence. Fail here with
        # an explanation rather than degrading to findings with no evidence.
        raise TokenizerError(
            f"{model!r} resolved to a slow (Python) tokenizer. kvlint needs a fast "
            "tokenizer for character offsets. Install `tokenizers`, or pick a model "
            "that ships a tokenizer.json."
        )
    return tokenizer


def encode(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Tokenize `text`, returning token ids and per-token character offsets.

    ASSUMPTION: ``add_special_tokens=False``. The chat template has already
    written BOS/EOS and role markers into the text as literal strings, so letting
    the tokenizer add its own would double them and inflate every token count
    against what the server reports.
    """
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    token_ids: list[int] = list(encoding["input_ids"])
    offsets: list[tuple[int, int]] = [(int(a), int(b)) for a, b in encoding["offset_mapping"]]

    if len(token_ids) != len(offsets):  # pragma: no cover
        raise TokenizerError(f"tokenizer returned {len(token_ids)} ids but {len(offsets)} offsets")
    return token_ids, offsets
