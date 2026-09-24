"""Turn bench/results/runs/*.json into bench/results/results.md and SVG charts.

Charts are plain SVG (no plotting dependency) with light and dark palettes via
`prefers-color-scheme`, a <title> tooltip on every mark, and the same numbers in
markdown tables next to them (the accessible view).

    python bench/report_bench.py
"""

from __future__ import annotations

import json
import statistics
import time
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "bench" / "results" / "runs"
OUT = ROOT / "bench" / "results"
LEVELS = {"basic": 0, "medium": 1, "hard": 2}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

# One hue, two validated shades (dataviz validator, --ordinal, light and dark surfaces):
# the baseline is the recessive shade, AutoCover-Lite the emphasised one.
STYLE = """<style>
  .bg { fill: #fcfcfb } .ink { fill: #0b0b0b } .ink2 { fill: #52514e } .muted { fill: #898781 }
  .grid { stroke: #e1e0d9 } .axis { stroke: #c3c2b7 }
  .base { fill: #86b6ef } .auto { fill: #256abf } .link { stroke: #c3c2b7 }
  .line-auto { stroke: #256abf } .line-base { stroke: #86b6ef } .ring { stroke: #fcfcfb }
  @media (prefers-color-scheme: dark) {
    .bg { fill: #1a1a19 } .ink { fill: #ffffff } .ink2 { fill: #c3c2b7 }
    .grid { stroke: #2c2c2a } .axis { stroke: #383835 }
    .base { fill: #184f95 } .auto { fill: #6da7ec } .link { stroke: #383835 }
    .line-auto { stroke: #6da7ec } .line-base { stroke: #184f95 } .ring { stroke: #1a1a19 }
  }
  text { font-family: FONT_FAMILY }
</style>""".replace("FONT_FAMILY", FONT)


def load() -> dict[str, dict[str, dict]]:
    by_subject: dict[str, dict[str, dict]] = {}
    for path in sorted(RUNS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        by_subject.setdefault(data["subject"], {})[data["tool"]] = data
    return dict(sorted(by_subject.items(),
                       key=lambda kv: (LEVELS.get(_any(kv[1]).get("level"), 9), kv[0])))


def _any(tools: dict) -> dict:
    return next(iter(tools.values()))


def metrics(result: dict | None) -> dict:
    """Uniform view of a baseline or AutoCover-Lite result."""
    if not result or result.get("error"):
        return {"ok": False, "error": (result or {}).get("error", "not run")}
    if result["tool"] == "autocover":
        final = result["final"]
        return {"ok": True, "line": final["line_pct"], "branch": final["branch_pct"],
                "mutation": (result.get("mutation") or {}).get("score_pct", 0.0),
                "tests": result["tests_in_suite"], "calls": result.get("llm_calls", 0),
                "tokens": result.get("llm_tokens", 0), "wall": result.get("wall_s", 0.0),
                "extra": f"{result['rounds']} rounds, stopped: {result['stopped_by']}"}
    return {"ok": True, "line": result["line_pct"], "branch": result["branch_pct"],
            "mutation": result["mutation"]["score_pct"], "tests": result["tests_passing"],
            "generated": result["tests_generated"], "calls": result["llm_calls"],
            "tokens": result["llm_tokens"], "wall": result["wall_s"],
            "extra": f"median of {len(result.get('samples', [1]))} samples"}


def coverage_at(curve: list[list[float]], seconds: float) -> float:
    reached = [p[1] for p in curve if p[0] <= seconds]
    return reached[-1] if reached else 0.0


# -- charts -------------------------------------------------------------------------------


def dumbbell_svg(rows: list[tuple[str, float, float]], title: str, unit: str = "%") -> str:
    """Baseline -> AutoCover-Lite per subject on a 0-100 axis."""
    left, right, top, row_h = 190, 40, 70, 34
    width = 760
    plot_w = width - left - right
    height = top + row_h * len(rows) + 50

    def x(v: float) -> float:
        return left + plot_w * max(0.0, min(100.0, v)) / 100

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="{width}" height="{height}" role="img" aria-label="{escape(title)}">',
           STYLE, f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
           f'<text class="ink" x="24" y="30" font-size="16" font-weight="600">'
           f'{escape(title)}</text>',
           f'<circle class="base" cx="{left}" cy="50" r="5"/>'
           f'<text class="ink2" x="{left + 10}" y="54" font-size="12">single-prompt baseline'
           f' (median of 3)</text>',
           f'<circle class="auto" cx="{left + 230}" cy="50" r="5"/>'
           f'<text class="ink2" x="{left + 240}" y="54" font-size="12">AutoCover-Lite</text>']
    for tick in range(0, 101, 25):
        out.append(f'<line class="grid" x1="{x(tick)}" x2="{x(tick)}" y1="{top - 6}" '
                   f'y2="{height - 40}" stroke-width="1"/>'
                   f'<text class="muted" x="{x(tick)}" y="{height - 22}" font-size="11" '
                   f'text-anchor="middle">{tick}{unit}</text>')
    for i, (label, base, auto) in enumerate(rows):
        cy = top + row_h * i + row_h / 2
        out.append(f'<text class="ink2" x="{left - 14}" y="{cy + 4}" font-size="12" '
                   f'text-anchor="end">{escape(label)}</text>')
        out.append(f'<line class="link" x1="{x(base)}" x2="{x(auto)}" y1="{cy}" y2="{cy}" '
                   f'stroke-width="2"/>')
        for cls, value, who in (("base", base, "baseline"), ("auto", auto, "AutoCover-Lite")):
            out.append(f'<circle class="{cls} ring" cx="{x(value)}" cy="{cy}" r="6" '
                       f'stroke-width="2">'
                       f'<title>{escape(label)} - {who}: {value}{unit}</title></circle>')
        lx = x(auto) + (12 if auto >= base else -12)
        anchor = "start" if auto >= base else "end"
        out.append(f'<text class="ink" x="{lx}" y="{cy + 4}" font-size="11" '
                   f'text-anchor="{anchor}">{auto:g}{unit}</text>')
    out.append("</svg>")
    return "\n".join(out)


def curves_svg(panels: list[tuple[str, list[list[float]], float, float]],
               budget_s: float) -> str:
    """Small multiples: AutoCover-Lite line coverage over time, baseline as reference."""
    cols, pw, ph, gap = 3, 240, 150, 26
    rows = (len(panels) + cols - 1) // cols
    width, top = cols * pw + (cols + 1) * gap, 60
    height = top + rows * (ph + 44) + 20
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
           f'width="{width}" height="{height}" role="img" '
           f'aria-label="Line coverage over time per subject">', STYLE,
           f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
           f'<text class="ink" x="{gap}" y="28" font-size="16" font-weight="600">'
           f'Line coverage over time (AutoCover-Lite), baseline dashed</text>']
    for i, (name, curve, base_line, final_line) in enumerate(panels):
        ox = gap + (i % cols) * (pw + gap)
        oy = top + (i // cols) * (ph + 44) + 20

        def px(t: float, _ox=ox) -> float:
            return _ox + pw * min(t, budget_s) / budget_s

        def py(v: float, _oy=oy) -> float:
            return _oy + ph - ph * v / 100

        out.append(f'<text class="ink2" x="{ox}" y="{oy - 8}" font-size="12" '
                   f'font-weight="600">{escape(name)}</text>')
        for v in (0, 50, 100):
            out.append(f'<line class="grid" x1="{ox}" x2="{ox + pw}" y1="{py(v)}" y2="{py(v)}" '
                       f'stroke-width="1"/><text class="muted" x="{ox - 4}" y="{py(v) + 3}" '
                       f'font-size="9" text-anchor="end">{v}</text>')
        out.append(f'<line class="axis" x1="{ox}" x2="{ox + pw}" y1="{py(0)}" y2="{py(0)}" '
                   f'stroke-width="1"/>')
        out.append(f'<text class="muted" x="{ox + pw}" y="{py(0) + 13}" font-size="9" '
                   f'text-anchor="end">{budget_s / 60:g} min</text>')
        out.append(f'<line class="line-base" x1="{ox}" x2="{ox + pw}" y1="{py(base_line)}" '
                   f'y2="{py(base_line)}" stroke-width="2" stroke-dasharray="5 4">'
                   f'<title>{escape(name)} baseline: {base_line}%</title></line>')
        points = [[0.0, 0.0]] + [[p[0], p[1]] for p in curve[1:]]
        path = " ".join(f"{'M' if j == 0 else 'L'}{px(t):.1f},{py(v):.1f}"
                        for j, (t, v) in enumerate(points))
        out.append(f'<path class="line-auto" d="{path}" fill="none" stroke-width="2" '
                   f'stroke-linejoin="round"><title>{escape(name)}: {final_line}% final'
                   f'</title></path>')
        t_end, v_end = points[-1]
        out.append(f'<circle class="auto" cx="{px(t_end)}" cy="{py(v_end)}" r="4"/>'
                   f'<text class="ink" x="{px(t_end) - 6}" y="{py(v_end) - 8}" font-size="10" '
                   f'text-anchor="end">{final_line:g}%</text>')
    out.append("</svg>")
    return "\n".join(out)


# -- markdown -----------------------------------------------------------------------------


def main() -> None:
    data = load()
    rows, dumb_mut, dumb_line, panels = [], [], [], []
    budget_min = max((_any(t).get("budget_min", 15) for t in data.values()), default=15)
    for name, tools in data.items():
        b, a = metrics(tools.get("baseline")), metrics(tools.get("autocover"))
        level = _any(tools).get("level", "")
        rows.append((name, level, b, a, tools.get("autocover")))
        if b["ok"] and a["ok"]:
            dumb_mut.append((name, b["mutation"], a["mutation"]))
            dumb_line.append((name, b["line"], a["line"]))
            panels.append((name, tools["autocover"].get("curve", []), b["line"], a["line"]))

    (OUT / "mutation_score.svg").write_text(
        dumbbell_svg(dumb_mut, "Mutation score (share of planted bugs caught)"), encoding="utf-8")
    (OUT / "line_coverage.svg").write_text(
        dumbbell_svg(dumb_line, "Line coverage"), encoding="utf-8")
    (OUT / "coverage_over_time.svg").write_text(curves_svg(panels, budget_min * 60),
                                               encoding="utf-8")

    both = [(b, a) for _, _, b, a, _ in rows if b["ok"] and a["ok"]]

    def mean(values):
        return round(statistics.mean(values), 1) if values else 0.0

    md = ["# Benchmark: AutoCover-Lite vs single-prompt baseline\n",
          f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `bench/results/runs/`. "
          f"{len(both)} of {len(rows)} subjects have results for both tools; "
          f"AutoCover-Lite budget {budget_min:g} min per subject.\n"]
    if both:
        md += ["## Headline (mean over subjects)\n",
               "| | Baseline | AutoCover-Lite |", "|---|---|---|",
               f"| Line coverage | {mean([b['line'] for b, _ in both])}% | "
               f"{mean([a['line'] for _, a in both])}% |",
               f"| Branch coverage | {mean([b['branch'] for b, _ in both])}% | "
               f"{mean([a['branch'] for _, a in both])}% |",
               f"| Mutation score | {mean([b['mutation'] for b, _ in both])}% | "
               f"{mean([a['mutation'] for _, a in both])}% |",
               f"| Tests kept | {mean([b['tests'] for b, _ in both])} | "
               f"{mean([a['tests'] for _, a in both])} |",
               f"| LLM calls | {mean([b['calls'] for b, _ in both])} | "
               f"{mean([a['calls'] for _, a in both])} |",
               f"| Wall time | {mean([b['wall'] for b, _ in both])}s | "
               f"{mean([a['wall'] for _, a in both])}s |", ""]
    md += ["![Mutation score](mutation_score.svg)\n", "![Line coverage](line_coverage.svg)\n",
           "![Coverage over time](coverage_over_time.svg)\n", "## Per subject\n",
           "| Subject | Level | Lines B -> A | Branches B -> A | Mutation B -> A | "
           "Tests B (passing/generated) | Tests A | LLM calls B / A | Time B / A | Notes |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, level, b, a, _ in rows:
        if not (b["ok"] and a["ok"]):
            md.append(f"| {name} | {level} | - | - | - | - | - | - | - | "
                      f"baseline: {b.get('error', 'ok')}; autocover: {a.get('error', 'ok')} |")
            continue
        md.append(f"| {name} | {level} | {b['line']} -> **{a['line']}** | "
                  f"{b['branch']} -> **{a['branch']}** | {b['mutation']} -> **{a['mutation']}** | "
                  f"{b['tests']}/{b.get('generated', '?')} | {a['tests']} | "
                  f"{b['calls']} / {a['calls']} | {b['wall']:.0f}s / {a['wall']:.0f}s | "
                  f"{a['extra']} |")
    md += ["", "## AutoCover-Lite coverage within shorter budgets\n",
           "Read off each run's coverage-over-time curve (the paper's Figure 2 method).\n",
           "| Subject | 5 min | 10 min | 15 min | Final |", "|---|---|---|---|---|"]
    for name, _, _, a, raw in rows:
        if a["ok"] and raw:
            curve = raw.get("curve", [])
            md.append(f"| {name} | {coverage_at(curve, 300)}% | {coverage_at(curve, 600)}% | "
                      f"{coverage_at(curve, 900)}% | {a['line']}% |")
    md += ["", "## How to read this\n",
           "- **Baseline**: the same Generator model chain and test-writing rules, one call "
           "for the whole module, failing tests dropped; the median of 3 samples is shown.",
           "- **Mutation score**: both tools are scored on the identical seeded mutant pool "
           "(`mutation.max_mutants_per_function` / `max_mutants_total`).",
           "- **Caveats**: one AutoCover-Lite run per subject; free-tier models answer "
           "differently depending on quotas and load (see the models used in "
           "`bench/results/runs/*.json`); subjects are pinned wheel versions with their own "
           "tests removed."]
    (OUT / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'results.md'} ({len(rows)} subjects)")


if __name__ == "__main__":
    main()
