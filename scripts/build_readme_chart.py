"""Generate the before/after chart the README leads with.

Run: `uv run python scripts/build_readme_chart.py`

Hand-built SVG rather than a plotly export, for two reasons: it needs no extra
dependency (a PNG export would pull in kaleido), and GitHub renders inline SVG
in a README while it will not render an interactive HTML chart.

The numbers come from actually running the demo pipeline, not from a constant in
this file. Regenerate it whenever the demo or the simulators change, so the chart
in the README can never drift away from what the tool actually produces.
"""

from __future__ import annotations

import pathlib

from kvlint.demo.synthetic import generate
from kvlint.fix.rewriter import fix_requests
from kvlint.lint.engine import run as run_rules
from kvlint.sim import sglang_radix, vllm_blocks
from kvlint.tokenize.renderer import tokenize_requests

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = pathlib.Path(__file__).resolve().parents[1] / "docs" / "before-after.svg"

WIDTH, HEIGHT = 720, 300
PLOT_LEFT, PLOT_TOP = 70, 60
PLOT_W, PLOT_H = 600, 180
BAR_W = 64


def measure() -> dict[str, object]:
    dirty = generate()
    before_tokens = tokenize_requests(MODEL, dirty)
    after_tokens = tokenize_requests(MODEL, fix_requests(dirty).requests)

    return {
        "vllm_before": vllm_blocks.simulate(before_tokens).hit_rate,
        "vllm_after": vllm_blocks.simulate(after_tokens).hit_rate,
        "sglang_before": sglang_radix.simulate(before_tokens).hit_rate,
        "sglang_after": sglang_radix.simulate(after_tokens).hit_rate,
        "findings": len(run_rules(dirty, before_tokens)),
        "requests": len(dirty),
    }


def bar(x: float, value: float, colour: str, label: str) -> str:
    height = value * PLOT_H
    y = PLOT_TOP + PLOT_H - height
    return (
        f'<rect x="{x:.0f}" y="{y:.1f}" width="{BAR_W}" height="{height:.1f}" '
        f'fill="{colour}" rx="3"/>\n'
        f'<text x="{x + BAR_W / 2:.0f}" y="{y - 8:.1f}" text-anchor="middle" '
        f'font-size="15" font-weight="600" fill="#222">{value * 100:.1f}%</text>\n'
        f'<text x="{x + BAR_W / 2:.0f}" y="{PLOT_TOP + PLOT_H + 20:.0f}" '
        f'text-anchor="middle" font-size="13" fill="#555">{label}</text>\n'
    )


def build(m: dict[str, object]) -> str:
    red, green = "#c0392b", "#27ae60"
    groups = [
        (150, "vLLM", float(m["vllm_before"]), float(m["vllm_after"])),
        (420, "SGLang", float(m["sglang_before"]), float(m["sglang_after"])),
    ]

    body = ""
    for x, name, before, after in groups:
        body += bar(x, before, red, "before")
        body += bar(x + BAR_W + 18, after, green, "after")
        delta = (after - before) * 100
        body += (
            f'<text x="{x + BAR_W + 9:.0f}" y="{PLOT_TOP + PLOT_H + 44:.0f}" '
            f'text-anchor="middle" font-size="14" font-weight="600" fill="#222">'
            f"{name} +{delta:.1f} pp</text>\n"
        )

    # Gridlines at 25% intervals, so the bars can be read without a tooltip.
    grid = ""
    for pct in (0, 25, 50, 75, 100):
        y = PLOT_TOP + PLOT_H - (pct / 100) * PLOT_H
        grid += (
            f'<line x1="{PLOT_LEFT}" y1="{y:.1f}" x2="{PLOT_LEFT + PLOT_W}" y2="{y:.1f}" '
            f'stroke="#e8e8e8" stroke-width="1"/>\n'
            f'<text x="{PLOT_LEFT - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-size="12" fill="#888">{pct}%</text>\n'
        )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}"
     viewBox="0 0 {WIDTH} {HEIGHT}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">
<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>
<text x="{PLOT_LEFT}" y="30" font-size="17" font-weight="600" fill="#222">
Prefix cache hit rate, before and after kvlint fix</text>
<text x="{PLOT_LEFT}" y="48" font-size="13" fill="#666">
{m["requests"]} synthetic support requests, {m["findings"]} issues found,
Qwen2.5-0.5B-Instruct tokenizer</text>
{grid}{body}</svg>
"""


def main() -> None:
    m = measure()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build(m), encoding="utf-8", newline="\n")

    print(f"wrote {OUT}")
    print(f"  vLLM   {float(m['vllm_before']) * 100:.1f}% -> {float(m['vllm_after']) * 100:.1f}%")
    print(
        f"  SGLang {float(m['sglang_before']) * 100:.1f}% -> {float(m['sglang_after']) * 100:.1f}%"
    )
    print(f"  findings: {m['findings']}")


if __name__ == "__main__":
    main()
