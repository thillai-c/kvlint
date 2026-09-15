"""Demo generator, JSON writer, HTML report, and the demo/report commands (M7)."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from kvlint.cli import ExitCode, app
from kvlint.demo.synthetic import generate
from kvlint.errors import KvlintError
from kvlint.fix.rewriter import fix_requests
from kvlint.lint.engine import run as run_rules
from kvlint.models import BeforeAfter, LintFinding, PerRequestResult, Report, SimResult
from kvlint.report import html, json_out
from kvlint.sim import vllm_blocks
from kvlint.sim.metrics import hit_rate_curve, suggest_budgets
from kvlint.tokenize.renderer import tokenize_requests
from tests.conftest import TINY_TOKENIZER_DIR

runner = CliRunner()
MODEL = str(TINY_TOKENIZER_DIR)


# ------------------------------------------------------------------ demo log


def test_demo_log_is_reproducible() -> None:
    """A demo whose headline number moves every run cannot be checked."""
    assert generate(count=10, seed=0) == generate(count=10, seed=0)


def test_demo_log_injects_all_three_anti_patterns() -> None:
    system = generate(count=2)[0].messages[0].content
    assert "Current time:" in system
    assert "Request id: req_" in system
    assert "@example.com" in system


def test_anti_patterns_sit_above_the_stable_instructions() -> None:
    """Position is the point. Below the instructions they would cost almost nothing."""
    system = generate(count=2)[0].messages[0].content
    assert system.index("Current time:") < system.index("You are Acme Corp")


def test_clean_mode_injects_nothing() -> None:
    system = generate(count=4, dirty=False)[0].messages[0].content
    assert system.startswith("You are Acme Corp")
    assert "Current time:" not in system


def test_the_demo_log_is_actually_broken() -> None:
    """Guards the demo itself: a log with no problems would make a pointless demo."""
    dirty = generate(count=20)
    findings = run_rules(dirty, tokenize_requests(MODEL, dirty))
    assert {"dynamic_time", "dynamic_id", "user_in_system"} <= {f.rule_id for f in findings}


def test_fixing_the_demo_log_clears_forty_points() -> None:
    dirty = generate(count=40)
    before = vllm_blocks.simulate(tokenize_requests(MODEL, dirty)).hit_rate
    after = vllm_blocks.simulate(tokenize_requests(MODEL, fix_requests(dirty).requests)).hit_rate
    assert (after - before) * 100 >= 40.0


# ------------------------------------------------------------------ json output


def sample_report() -> Report:
    sim = SimResult(
        engine="vllm",
        config={"block_size": 16},
        per_request=[
            PerRequestResult(request_id="r0", prompt_tokens=100, cached_tokens=0),
            PerRequestResult(
                request_id="r1", prompt_tokens=100, cached_tokens=64, divergence_token_idx=64
            ),
        ],
        hit_rate=0.32,
        ideal_hit_rate=0.32,
        block_size=16,
    )
    return Report(
        input_summary={"path": "x.jsonl", "format": "openai", "model": "m", "requests": 2},
        sims=[sim],
        findings=[
            LintFinding(
                rule_id="dynamic_time",
                severity="high",
                affected_request_ids=["r1"],
                affected_fraction=0.5,
                est_tokens_lost=36,
                evidence="<script>alert(1)</script>",
                suggestion="Move the timestamp down.",
                auto_fixable=True,
            )
        ],
        before_after=BeforeAfter(
            hit_rate_before=0.32,
            hit_rate_after=0.80,
            per_engine={"vllm": {"before": 0.32, "after": 0.80}},
        ),
        version="0.0.1",
    )


def test_json_output_is_byte_identical_across_writes(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    json_out.write(a, sample_report())
    json_out.write(b, sample_report())
    assert a.read_bytes() == b.read_bytes()


def test_json_output_uses_lf_on_every_platform(tmp_path: Path) -> None:
    """A report differing only by line endings would defeat the guarantee."""
    out = tmp_path / "r.json"
    json_out.write(out, sample_report())
    assert b"\r\n" not in out.read_bytes()


def test_json_output_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "r.json"
    json_out.write(out, sample_report())
    assert Report.model_validate(json.loads(out.read_text(encoding="utf-8"))) == sample_report()


# ------------------------------------------------------------------ html report


def test_html_contains_all_three_charts(tmp_path: Path) -> None:
    out = tmp_path / "r.html"
    html.write(out, sample_report(), curve=[(10, 0.2), (20, 0.5)])
    text = out.read_text(encoding="utf-8")

    assert "before and after" in text
    assert "Where prompts stop matching" in text
    assert "Hit rate against KV budget" in text


def test_html_is_self_contained(tmp_path: Path) -> None:
    """Bundling plotly means the file opens offline, which matters for CI artifacts."""
    out = tmp_path / "r.html"
    html.write(out, sample_report())
    assert "plotly" in out.read_text(encoding="utf-8").lower()


def test_html_escapes_finding_text() -> None:
    """Evidence is quoted from user prompts, so it cannot be trusted as markup."""
    rendered = html.render(sample_report())
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered


def test_html_omits_charts_it_has_no_data_for() -> None:
    empty = Report(input_summary={"requests": 0}, sims=[], findings=[], version="0.0.1")
    rendered = html.render(empty)
    assert "Hit rate against KV budget" not in rendered
    assert "No cache-breaking patterns found" in rendered


def test_missing_plotly_gives_an_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """M7 acceptance: the optional extra degrades with a message, not a traceback."""
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    with pytest.raises(KvlintError, match="kvlint\\[report\\]"):
        html.render(sample_report())


# ------------------------------------------------------------------ budget curve


def test_budget_curve_rises_as_memory_grows() -> None:
    dirty = generate(count=20)
    tokenized = tokenize_requests(MODEL, dirty)
    curve = hit_rate_curve(tokenized, suggest_budgets(tokenized))

    assert len(curve) >= 2
    rates = [rate for _, rate in curve]
    assert rates == sorted(rates), "more KV memory must never lower the hit rate"


def test_suggest_budgets_handles_a_tiny_log() -> None:
    assert suggest_budgets(tokenize_requests(MODEL, generate(count=1))) != [0]


# ------------------------------------------------------------------ commands


def test_demo_runs_with_no_arguments() -> None:
    """M7 acceptance, using the vendored tokenizer so the test stays offline."""
    result = runner.invoke(app, ["demo", "--model", MODEL, "--requests", "20"])

    assert result.exit_code == ExitCode.OK, result.output
    flat = re.sub(r"\s+", " ", result.output)
    assert "Hit rate" in flat
    assert "after fixing" in flat


def test_demo_headline_shows_a_real_improvement() -> None:
    result = runner.invoke(app, ["demo", "--model", MODEL, "--requests", "40"])
    flat = re.sub(r"\s+", " ", result.output)

    match = re.search(r"Hit rate ([\d.]+)% to ([\d.]+)%", flat)
    assert match, flat
    before, after = float(match.group(1)), float(match.group(2))
    assert after - before >= 40.0


def test_demo_writes_a_log_the_ingester_can_read(tmp_path: Path) -> None:
    from kvlint.ingest import load as load_log

    out = tmp_path / "demo.jsonl"
    runner.invoke(app, ["demo", "--model", MODEL, "--requests", "10", "--out", str(out)])
    assert len(load_log(out, "openai")) == 10


def test_demo_writes_html(tmp_path: Path) -> None:
    out = tmp_path / "demo.html"
    result = runner.invoke(app, ["demo", "--model", MODEL, "--requests", "10", "--html", str(out)])
    assert result.exit_code == ExitCode.OK
    assert out.exists() and out.stat().st_size > 1000


def test_report_renders_a_saved_json_file(tmp_path: Path) -> None:
    source, out = tmp_path / "r.json", tmp_path / "r.html"
    json_out.write(source, sample_report())

    result = runner.invoke(app, ["report", str(source), "--html", str(out)])
    assert result.exit_code == ExitCode.OK
    assert "Where prompts stop matching" in out.read_text(encoding="utf-8")


def test_report_rejects_a_missing_file(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["report", str(tmp_path / "nope.json"), "--html", str(tmp_path / "o.html")]
    )
    assert result.exit_code == ExitCode.ERROR
    assert "file not found" in result.output


def test_report_rejects_a_non_report_json(tmp_path: Path) -> None:
    source = tmp_path / "other.json"
    source.write_text('{"unrelated": true}', encoding="utf-8")

    result = runner.invoke(app, ["report", str(source), "--html", str(tmp_path / "o.html")])
    assert result.exit_code == ExitCode.ERROR
    assert "not a kvlint report" in result.output


def test_report_rejects_malformed_json(tmp_path: Path) -> None:
    source = tmp_path / "bad.json"
    source.write_text("{not json", encoding="utf-8")

    result = runner.invoke(app, ["report", str(source), "--html", str(tmp_path / "o.html")])
    assert result.exit_code == ExitCode.ERROR
    assert "not valid JSON" in result.output


def test_analyze_writes_html(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(
        "\n".join(
            json.dumps({"messages": [{"role": "user", "content": f"question {i} about orders"}]})
            for i in range(6)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "a.html"
    result = runner.invoke(app, ["analyze", str(log), "--model", MODEL, "--html", str(out)])
    assert result.exit_code == ExitCode.OK, result.output
    assert out.exists()


def test_no_command_prints_the_transformers_banner(tmp_path: Path) -> None:
    """The import banner is noise in CLI output and was suppressed in M7."""
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"messages": [{"role": "user", "content": "hi"}]}) + "\n", encoding="utf-8"
    )

    result = runner.invoke(app, ["analyze", str(log), "--model", MODEL])
    assert "PyTorch was not found" not in result.output


def test_error_messages_survive_rich_markup(tmp_path: Path) -> None:
    """Regression: rich ate the brackets out of the plotly install hint.

    `pip install 'kvlint[report]'` printed as `pip install 'kvlint'`, because
    rich read `[report]` as a style tag. That is advice that does not work, and
    the same hazard applies to any error quoting a path or a prompt excerpt.
    """
    source = tmp_path / "weird[1].json"
    source.write_text("{not json", encoding="utf-8")
    result = runner.invoke(app, ["report", str(source), "--html", str(tmp_path / "o.html")])

    assert result.exit_code == ExitCode.ERROR
    assert "weird[1].json" in re.sub(r"\s+", " ", result.output)


def test_the_plotly_hint_keeps_its_brackets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact string that regressed: the extra name must reach the user intact."""
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)

    source = tmp_path / "r.json"
    json_out.write(source, sample_report())
    result = runner.invoke(app, ["report", str(source), "--html", str(tmp_path / "o.html")])

    assert result.exit_code == ExitCode.ERROR
    assert "kvlint[report]" in re.sub(r"\s+", " ", result.output)


# ------------------------------------------------------------------ chart quality


def test_the_demo_workload_has_competing_prefixes() -> None:
    """Several tenants, not one shared system prompt.

    With a single shared prefix, those blocks are touched by every request and so
    are always the most recently used. They can never be evicted, the hit rate is
    identical at every KV budget, and the memory-pressure chart is a flat line
    that wrongly reads as "memory does not matter here".
    """
    from kvlint.demo.synthetic import TENANTS

    systems = [r.messages[0].content for r in generate(count=40, dirty=False)]
    assert len(set(systems)) == len(TENANTS)
    assert not any("{tenant}" in s for s in systems), "template placeholder left unformatted"


def test_the_budget_curve_actually_moves() -> None:
    """The eviction axis has to be visible, or the chart teaches nothing."""
    fixed = fix_requests(generate(count=60)).requests
    tokenized = tokenize_requests(MODEL, fixed)
    rates = [rate for _, rate in hit_rate_curve(tokenized, suggest_budgets(tokenized))]

    assert max(rates) - min(rates) > 0.2, "no knee: eviction never bites on this workload"


def test_budgets_are_log_spaced_from_one() -> None:
    """Linear spacing from total/8 steps straight over the knee."""
    budgets = suggest_budgets(tokenize_requests(MODEL, generate(count=40)))

    assert budgets[0] == 1
    # Each step multiplies rather than adds, so early budgets stay close together.
    assert budgets[1] - budgets[0] < budgets[-1] - budgets[-2]


def test_divergence_chart_narrows_buckets_when_positions_cluster() -> None:
    """One bar spanning the plot looks broken; it should read as a finding."""
    dirty = generate(count=30)
    tokenized = tokenize_requests(MODEL, dirty)
    report = Report(
        input_summary={"requests": len(dirty)},
        sims=[vllm_blocks.simulate(tokenized)],
        findings=[],
        version="0.0.1",
    )
    rendered = html.render(report)
    assert "Every request diverges at token" in rendered
