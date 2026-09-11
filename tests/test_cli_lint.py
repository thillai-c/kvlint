"""End-to-end `lint` tests, including the CI exit-code gate."""

from __future__ import annotations

import json
import re
from pathlib import Path

from typer.testing import CliRunner

from kvlint.cli import ExitCode, app
from tests.conftest import TINY_TOKENIZER_DIR

runner = CliRunner()

MODEL = str(TINY_TOKENIZER_DIR)
SYSTEM = (
    "You are a support assistant for Acme Corp. "
    "Answer concisely and always cite the knowledge base article you used."
)


def write_log(path: Path, *, timestamps: bool, count: int = 12) -> Path:
    lines = []
    for i in range(count):
        system = SYSTEM
        if timestamps:
            system += f"\nCurrent time: 2026-09-11T08:{i:02d}:00Z"
        lines.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": f"Where is order {chr(ord('A') + i)}?"},
                    ]
                }
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_lint_reports_an_injected_timestamp(tmp_path: Path) -> None:
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    result = runner.invoke(app, ["lint", str(log), "--model", MODEL])

    assert result.exit_code == ExitCode.OK, result.output
    assert "dynamic_time" in result.output
    assert "Findings" in result.output


def test_lint_shows_hit_rates_for_both_engines(tmp_path: Path) -> None:
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    result = runner.invoke(app, ["lint", str(log), "--model", MODEL])
    assert "vLLM hit rate" in result.output
    assert "SGLang" in result.output


def test_fail_on_high_exits_two(tmp_path: Path) -> None:
    """Exit code 2 is what makes this usable as a CI gate (build plan section 6)."""
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    result = runner.invoke(app, ["lint", str(log), "--model", MODEL, "--fail-on", "high"])
    assert result.exit_code == ExitCode.FINDINGS


def test_without_fail_on_a_dirty_log_still_exits_zero(tmp_path: Path) -> None:
    """Reporting is not failing unless the user asked for a gate."""
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    result = runner.invoke(app, ["lint", str(log), "--model", MODEL])
    assert result.exit_code == ExitCode.OK


def test_fail_on_low_does_not_trigger_without_findings(tmp_path: Path) -> None:
    log = tmp_path / "single.jsonl"
    log.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hello"}]}) + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["lint", str(log), "--model", MODEL, "--fail-on", "low"])
    assert result.exit_code == ExitCode.OK


def test_lint_writes_findings_into_the_json_report(tmp_path: Path) -> None:
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    out = tmp_path / "report.json"
    runner.invoke(app, ["lint", str(log), "--model", MODEL, "--json", str(out)])

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert any(f["rule_id"] == "dynamic_time" for f in payload["findings"])
    assert len(payload["sims"]) == 2


def test_lint_json_report_is_byte_identical_across_runs(tmp_path: Path) -> None:
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for out in (first, second):
        runner.invoke(app, ["lint", str(log), "--model", MODEL, "--json", str(out)])
    assert first.read_bytes() == second.read_bytes()


def test_verbose_prints_the_suggestion(tmp_path: Path) -> None:
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    result = runner.invoke(app, ["lint", str(log), "--model", MODEL, "--verbose"])
    assert "Move the timestamp" in result.output


def test_missing_log_exits_with_an_error(tmp_path: Path) -> None:
    result = runner.invoke(app, ["lint", str(tmp_path / "nope.jsonl"), "--model", MODEL])
    assert result.exit_code == ExitCode.ERROR
    assert "file not found" in result.output


def test_totals_are_labelled_as_non_additive(tmp_path: Path) -> None:
    """Rules overlap, so the tokens-lost column must not read as a sum."""
    log = write_log(tmp_path / "dirty.jsonl", timestamps=True)
    result = runner.invoke(app, ["lint", str(log), "--model", MODEL])
    # rich hard-wraps to the terminal width, so compare against unwrapped text.
    flattened = re.sub(r"\s+", " ", result.output)
    assert "does not sum" in flattened
