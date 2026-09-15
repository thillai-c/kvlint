"""Shared test fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TINY_TOKENIZER_DIR = FIXTURES / "tiny_tokenizer"

# The real model the network-marked tests and M6 target.
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Wide enough that rich never hard-wraps a line during tests.
TEST_CONSOLE_WIDTH = "400"


@pytest.fixture(autouse=True, scope="session")
def _pin_console_width() -> Iterator[None]:
    """Stop rich from hard-wrapping CLI output during tests.

    Rich wraps to the terminal width, defaulting to 80 columns when there is no
    TTY. That means a message can be split across lines *depending on how long
    the preceding text is*, and `tmp_path` differs per platform:

        /tmp/pytest-of-runner/...         (Linux, ~75 chars)
        C:\\Users\\...\\AppData\\Local\\Temp\\... (Windows, ~125 chars)

    So `"file not found" in result.output` passed on Windows and macOS and failed
    on Linux, because only there did the wrap land mid-phrase. The wrap point is
    an accident of path length, never something a test should depend on.

    Wrapping itself is correct behaviour for a real terminal, so this pins the
    width for tests rather than changing what users see.
    """
    previous = os.environ.get("COLUMNS")
    os.environ["COLUMNS"] = TEST_CONSOLE_WIDTH
    yield
    if previous is None:
        os.environ.pop("COLUMNS", None)
    else:
        os.environ["COLUMNS"] = previous


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
