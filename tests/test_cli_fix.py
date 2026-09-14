"""End-to-end `fix` tests, including the M5 improvement criterion."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from kvlint.cli import ExitCode, app
from kvlint.ingest import load as load_log
from tests.conftest import TINY_TOKENIZER_DIR

runner = CliRunner()
MODEL = str(TINY_TOKENIZER_DIR)

STABLE = (
    "You are Acme Corp's customer support assistant.\n"
    "Always answer in two sentences or fewer.\n"
    "Cite the knowledge base article id you relied on, in square brackets.\n"
    "Never speculate about delivery dates you cannot verify from the order record.\n"
    "If the customer is angry, acknowledge the frustration before answering.\n"
    "Escalate to a human when the customer asks for a refund above 500 dollars."
)


def write_dirty_log(path: Path, count: int = 40) -> Path:
    """The realistic bad case: dynamic values above the stable instructions."""
    lines = []
    for i in range(count):
        system = (
            f"Current time: 2026-09-11T08:{i:02d}:00Z\n"
            f"Request id: req_{i:08d}f3\n"
            f"user: person{i}@example.com\n" + STABLE
        )
        lines.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Where is my order number {1000 + i}?"},
                    ]
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def change_in_pp(output: str, engine: str) -> float:
    """Pull the signed percentage-point change for one engine out of the table.

    Matched on content rather than on column separators, because rich draws the
    table with Unicode box characters that vary with terminal capabilities.
    """
    flat = re.sub(r"\s+", " ", output)
    match = re.search(rf"{engine}\D*?[\d.]+%\D*?[\d.]+%\D*?([+-][\d.]+) pp", flat)
    assert match, f"no {engine} row in:\n{output}"
    return float(match.group(1))


def test_fix_improves_the_hit_rate_by_at_least_40_points(tmp_path: Path) -> None:
    """M5 acceptance: a dirty log gains 40 percentage points or more."""
    log = write_dirty_log(tmp_path / "dirty.jsonl")
    result = runner.invoke(app, ["fix", str(log), "--model", MODEL])

    assert result.exit_code == ExitCode.OK, result.output
    assert change_in_pp(result.output, "vllm") >= 40.0
    assert change_in_pp(result.output, "sglang") >= 40.0


def test_fix_never_makes_a_clean_log_worse(tmp_path: Path) -> None:
    """A fixer that can regress is worse than no fixer."""
    log = tmp_path / "clean.jsonl"
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": STABLE},
                    {"role": "user", "content": f"Where is my order number {1000 + i}?"},
                ]
            }
        )
        for i in range(20)
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["fix", str(log), "--model", MODEL])
    assert change_in_pp(result.output, "vllm") >= 0.0
    assert change_in_pp(result.output, "sglang") >= 0.0
    assert "Rewrote 0 of 20" in result.output


def test_trailing_dynamic_lines_are_not_churned(tmp_path: Path) -> None:
    """Regression guard: relocating an already-trailing line lost 2.3 pp."""
    log = tmp_path / "trailing.jsonl"
    lines = [
        json.dumps(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": STABLE + f"\nCurrent time: 2026-09-11T08:{i:02d}:00Z",
                    },
                    {"role": "user", "content": f"Where is my order {i}?"},
                ]
            }
        )
        for i in range(20)
    ]
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["fix", str(log), "--model", MODEL])
    assert change_in_pp(result.output, "vllm") >= 0.0


def test_apply_writes_a_log_the_ingester_can_read(tmp_path: Path) -> None:
    """M5 acceptance: round-trip through the same ingester."""
    log = write_dirty_log(tmp_path / "dirty.jsonl", count=5)
    out = tmp_path / "fixed.jsonl"
    result = runner.invoke(app, ["fix", str(log), "--model", MODEL, "--apply", "--out", str(out)])

    assert result.exit_code == ExitCode.OK
    assert len(load_log(out, "openai")) == 5


def test_relinting_the_fixed_log_shows_a_much_cheaper_problem(tmp_path: Path) -> None:
    """Rerunning lint on the output must show the same rules costing far less.

    The rules keep firing, and that is correct: the timestamp is still a value
    that changes between requests. What the fix buys is position. It now sits
    below the stable instructions, so a miss costs the tail of the prompt rather
    than all of it. Asserting the findings vanish would be asserting a lie.
    """
    log = write_dirty_log(tmp_path / "dirty.jsonl", count=20)
    out = tmp_path / "fixed.jsonl"
    runner.invoke(app, ["fix", str(log), "--model", MODEL, "--apply", "--out", str(out)])

    for source, target in ((log, tmp_path / "b.json"), (out, tmp_path / "a.json")):
        result = runner.invoke(app, ["lint", str(source), "--model", MODEL, "--json", str(target)])
        assert result.exit_code == ExitCode.OK

    def tokens_lost(path: Path) -> int:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return max(
            (f["est_tokens_lost"] for f in payload["findings"] if f["severity"] == "high"),
            default=0,
        )

    assert tokens_lost(tmp_path / "a.json") < tokens_lost(tmp_path / "b.json") / 2


def test_apply_without_out_is_rejected(tmp_path: Path) -> None:
    log = write_dirty_log(tmp_path / "dirty.jsonl", count=3)
    result = runner.invoke(app, ["fix", str(log), "--model", MODEL, "--apply"])
    assert result.exit_code == ExitCode.ERROR
    assert "--out" in result.output


def test_estimates_are_labelled_as_estimates(tmp_path: Path) -> None:
    """Build plan section 5.5 requires this wherever they appear."""
    log = write_dirty_log(tmp_path / "dirty.jsonl", count=10)
    result = runner.invoke(app, ["fix", str(log), "--model", MODEL, "--price-per-1m-input", "0.30"])
    flat = re.sub(r"\s+", " ", result.output)
    assert "estimates" in flat
    assert "not measurements" in flat


def test_cost_is_absent_without_a_price(tmp_path: Path) -> None:
    log = write_dirty_log(tmp_path / "dirty.jsonl", count=5)
    result = runner.invoke(app, ["fix", str(log), "--model", MODEL])
    assert "cost change" not in result.output


def test_user_start_target_is_accepted(tmp_path: Path) -> None:
    log = write_dirty_log(tmp_path / "dirty.jsonl", count=10)
    result = runner.invoke(app, ["fix", str(log), "--model", MODEL, "--target", "user_start"])
    assert result.exit_code == ExitCode.OK
    assert change_in_pp(result.output, "sglang") >= 40.0


def test_missing_log_exits_with_an_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["fix", str(tmp_path / "nope.jsonl"), "--model", MODEL])
    assert result.exit_code == ExitCode.ERROR
    assert "file not found" in result.output


def test_verbose_reports_what_was_rewritten(tmp_path: Path) -> None:
    log = write_dirty_log(tmp_path / "dirty.jsonl", count=5)
    result = runner.invoke(app, ["fix", str(log), "--model", MODEL, "--verbose"])
    assert "volatile lines" in result.output
