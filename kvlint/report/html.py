"""Standalone HTML report.

Three charts, each answering a question the terminal output can only gesture at:

- **Before and after** per engine: what the fix is worth.
- **Divergence position**: where prompts stop agreeing. A spike near zero is the
  expensive case, because everything after it is recomputed.
- **Hit rate against KV budget**: whether the workload is limited by prompt
  construction or by memory. A curve that is already flat at the left means more
  GPU will not help.

Plotly is an optional extra. Importing it at module import time would make
`kvlint analyze` fail for everyone who did not install `[report]`, so the import
lives inside the function and raises a message that says what to install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kvlint.errors import KvlintError
from kvlint.models import Report
from kvlint.sim.metrics import divergence_histogram

PLOTLY_HINT = (
    "HTML reports need plotly. Install it with: pip install 'kvlint[report]' "
    "(or uv sync --all-extras)"
)


def _require_plotly() -> Any:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise KvlintError(PLOTLY_HINT) from exc
    return go


def _before_after_figure(go: Any, report: Report) -> Any | None:
    if report.before_after is None:
        return None

    engines = sorted(report.before_after.per_engine)
    figure = go.Figure(
        data=[
            go.Bar(
                name="Before",
                x=engines,
                y=[report.before_after.per_engine[e]["before"] * 100 for e in engines],
                marker_color="#c0392b",
            ),
            go.Bar(
                name="After",
                x=engines,
                y=[report.before_after.per_engine[e]["after"] * 100 for e in engines],
                marker_color="#27ae60",
            ),
        ]
    )
    figure.update_layout(
        title="Prefix cache hit rate, before and after fixes",
        yaxis_title="Hit rate (%)",
        yaxis_range=[0, 100],
        barmode="group",
        template="plotly_white",
    )
    return figure


def _divergence_figure(go: Any, report: Report, bucket_size: int = 64) -> Any | None:
    if not report.sims:
        return None

    # Size the buckets from the data rather than fixing them at 64. A workload
    # where every prompt breaks at the same token is common and important, but
    # with a fixed bucket it renders as one bar spanning the whole plot, which
    # looks like a broken chart instead of a finding.
    positions = [
        record.divergence_token_idx
        for sim in report.sims
        for record in sim.per_request
        if record.divergence_token_idx is not None
    ]
    if not positions:
        return None

    span = max(positions) - min(positions)
    bucket = bucket_size if span > bucket_size * 2 else max(1, span // 8 or 1)

    figure = go.Figure()
    for sim in report.sims:
        histogram = divergence_histogram(sim, bucket)
        if not histogram:
            continue
        labels = [
            str(start) if bucket == 1 else f"{start}-{start + bucket - 1}" for start in histogram
        ]
        figure.add_trace(go.Bar(name=sim.engine, x=labels, y=list(histogram.values())))
    if not figure.data:
        return None

    subtitle = ""
    if len(set(positions)) == 1:
        # Worth saying outright: one position means a single template defect, not
        # scattered drift, and it is the easiest kind to fix.
        subtitle = f"<br><sub>Every request diverges at token {positions[0]}</sub>"

    figure.update_layout(
        title=f"Where prompts stop matching (token position){subtitle}",
        xaxis_title="Token position of divergence",
        yaxis_title="Requests",
        barmode="group",
        bargap=0.6,
        template="plotly_white",
    )
    return figure


def _budget_figure(go: Any, curve: list[tuple[int, float]]) -> Any | None:
    """Hit rate as the KV budget shrinks, if the caller measured one."""
    if not curve:
        return None

    figure = go.Figure(
        data=[
            go.Scatter(
                x=[blocks for blocks, _ in curve],
                y=[rate * 100 for _, rate in curve],
                mode="lines+markers",
                line={"color": "#2980b9"},
            )
        ]
    )
    knee = next(
        (blocks for blocks, rate in curve if rate >= max(r for _, r in curve) * 0.95),
        None,
    )
    subtitle = f"<br><sub>Reaches its ceiling at about {knee:,} blocks</sub>" if knee else ""

    figure.update_layout(
        title=f"Hit rate against KV budget (vLLM){subtitle}",
        xaxis_title="KV budget (blocks)",
        yaxis_title="Hit rate (%)",
        yaxis_range=[0, 100],
        template="plotly_white",
    )
    return figure


def _findings_table(report: Report) -> str:
    if not report.findings:
        return "<p>No cache-breaking patterns found.</p>"

    # Every cell is escaped. Evidence and suggestion quote text taken straight
    # from user prompts, so neither can be trusted as markup.
    rows = "\n".join(
        "<tr>"
        f"<td class='sev-{_escape(f.severity)}'>{_escape(f.severity)}</td>"
        f"<td><code>{_escape(f.rule_id)}</code></td>"
        f"<td>{f.affected_fraction * 100:.1f}%</td>"
        f"<td>{f.est_tokens_lost:,}</td>"
        f"<td>{'auto' if f.auto_fixable else 'manual'}</td>"
        f"<td><code>{_escape(f.evidence)}</code></td>"
        f"<td>{_escape(f.suggestion)}</td>"
        "</tr>"
        for f in report.findings
    )
    return f"""<table>
<thead><tr><th>Severity</th><th>Rule</th><th>Affected</th>
<th>Tokens lost</th><th>Fix</th><th>Evidence</th><th>Suggestion</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="note">Tokens lost is an upper bound and rules can overlap,
so the column does not sum.</p>"""


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


STYLE = """
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif;
       max-width: 1000px; margin: 2rem auto; padding: 0 1rem; color: #222; }
h1 { margin-bottom: 0.2rem; }
.subtitle { color: #666; margin-top: 0; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
th, td { text-align: left; padding: 0.5rem; border-bottom: 1px solid #eee; }
th { background: #fafafa; }
code { background: #f4f4f4; padding: 0.1rem 0.3rem; border-radius: 3px; }
.sev-high { color: #c0392b; font-weight: 600; }
.sev-medium { color: #d68910; font-weight: 600; }
.sev-low { color: #888; }
.note { color: #666; font-size: 0.9rem; }
"""


def render(report: Report, curve: list[tuple[int, float]] | None = None) -> str:
    """Build the full HTML document."""
    go = _require_plotly()

    figures = [
        _before_after_figure(go, report),
        _divergence_figure(go, report),
        _budget_figure(go, curve or []),
    ]

    blocks = []
    for index, figure in enumerate(figures):
        if figure is None:
            continue
        # Bundle plotly.js once, then reference it, so the file works offline
        # without a CDN and does not carry three copies of the library.
        blocks.append(figure.to_html(full_html=False, include_plotlyjs=(index == 0)))

    summary = report.input_summary
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kvlint report</title>
<style>{STYLE}</style>
</head>
<body>
<h1>kvlint report</h1>
<p class="subtitle">{summary.get("requests", "?")} requests from
<code>{_escape(str(summary.get("path", "?")))}</code>,
model <code>{_escape(str(summary.get("model", "?")))}</code></p>
{"".join(blocks)}
<h2>Findings</h2>
{_findings_table(report)}
<p class="note">Generated by kvlint {report.version}. Hit rates are simulated;
TTFT and cost figures, where shown, are estimates from user-supplied rates.</p>
</body>
</html>
"""


def write(path: Path, report: Report, curve: list[tuple[int, float]] | None = None) -> None:
    path.write_text(render(report, curve), encoding="utf-8", newline="\n")
