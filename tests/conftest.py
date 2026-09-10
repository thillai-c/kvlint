"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TINY_TOKENIZER_DIR = FIXTURES / "tiny_tokenizer"

# The real model the network-marked tests and M6 target.
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture(scope="session")
def tiny_tokenizer() -> Any:
    """A vendored byte-level tokenizer: one byte per token, no network needed.

    Predictable token counts make the simulator assertions in M2/M3 checkable by
    hand, which is the whole reason this exists rather than a real tokenizer.
    """
    from kvlint.tokenize.tokenizer import load_tokenizer

    return load_tokenizer(str(TINY_TOKENIZER_DIR))
