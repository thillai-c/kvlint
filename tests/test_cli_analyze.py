"""End-to-end `analyze` tests (M3 acceptance: side-by-side engine table)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from kvlint.cli import ExitCode, app
from tests.conftest import TINY_TOKENIZER_DIR

runner = CliRunner()


def write_log(path: Path) -> Path:
    """Three requests sharing a long system prompt, so reuse is visible."""
    system = "You are a careful assistant. " * 8
    lines = [
        json.dumps(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Question number {i}"},
                ]
            }
        )
        for i in range(3)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_analyze_prints_both_engines(tmp_path: Path) -> None:
    log = write_log(tmp_path / "log.jsonl")
    result = runner.invoke(app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR)])

    assert result.exit_code == ExitCode.OK, result.output
    assert "vllm" in result.output
    assert "sglang" in result.output
    assert "Hit rate" in result.output


def test_analyze_can_select_one_engine(tmp_path: Path) -> None:
    log = write_log(tmp_path / "log.jsonl")
    result = runner.invoke(
        app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR), "--engine", "vllm"]
    )
    assert result.exit_code == ExitCode.OK
    assert "sglang" not in result.output


def test_analyze_writes_a_json_report(tmp_path: Path) -> None:
    log = write_log(tmp_path / "log.jsonl")
    out = tmp_path / "report.json"
    result = runner.invoke(
        app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR), "--json", str(out)]
    )

    assert result.exit_code == ExitCode.OK
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert {sim["engine"] for sim in payload["sims"]} == {"vllm", "sglang"}
    assert payload["input_summary"]["requests"] == 3


def test_json_report_is_byte_identical_across_runs(tmp_path: Path) -> None:
    """Build plan section 2: same input and config, byte-identical report."""
    log = write_log(tmp_path / "log.jsonl")
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for out in (first, second):
        runner.invoke(
            app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR), "--json", str(out)]
        )
    assert first.read_bytes() == second.read_bytes()


def test_sglang_reports_at_least_as_much_reuse_as_vllm(tmp_path: Path) -> None:
    log = write_log(tmp_path / "log.jsonl")
    out = tmp_path / "report.json"
    runner.invoke(
        app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR), "--json", str(out)]
    )
    sims = {s["engine"]: s for s in json.loads(out.read_text(encoding="utf-8"))["sims"]}
    assert sims["sglang"]["hit_rate"] >= sims["vllm"]["hit_rate"]


def test_verbose_adds_the_divergence_table(tmp_path: Path) -> None:
    log = write_log(tmp_path / "log.jsonl")
    result = runner.invoke(
        app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR), "--verbose"]
    )
    assert "Divergence position" in result.output


def test_missing_log_exits_with_an_error(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["analyze", str(tmp_path / "nope.jsonl"), "--model", str(TINY_TOKENIZER_DIR)]
    )
    assert result.exit_code == ExitCode.ERROR
    assert "file not found" in result.output


def test_plain_format_is_accepted(tmp_path: Path) -> None:
    log = tmp_path / "plain.jsonl"
    log.write_text(
        '{"prompt":"The quick brown fox jumps"}\n{"prompt":"The quick brown dog jumps"}\n',
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR), "--format", "plain"]
    )
    assert result.exit_code == ExitCode.OK


def test_kv_budget_reduces_the_hit_rate(tmp_path: Path) -> None:
    """The memory-pressure axis, visible end to end."""
    log = write_log(tmp_path / "log.jsonl")
    roomy, tight = tmp_path / "roomy.json", tmp_path / "tight.json"

    runner.invoke(
        app, ["analyze", str(log), "--model", str(TINY_TOKENIZER_DIR), "--json", str(roomy)]
    )
    runner.invoke(
        app,
        [
            "analyze",
            str(log),
            "--model",
            str(TINY_TOKENIZER_DIR),
            "--kv-budget-blocks",
            "2",
            "--kv-budget-tokens",
            "32",
            "--json",
            str(tight),
        ],
    )

    def hit_rates(path: Path) -> dict[str, float]:
        sims = json.loads(path.read_text(encoding="utf-8"))["sims"]
        return {s["engine"]: s["hit_rate"] for s in sims}

    for engine, rate in hit_rates(tight).items():
        assert rate <= hit_rates(roomy)[engine]
