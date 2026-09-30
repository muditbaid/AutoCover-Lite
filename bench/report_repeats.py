"""Summarise repeated AutoCover-Lite runs: mean and range per subject.

One run per subject cannot rank pipeline changes on free tiers (single subjects swung by
up to 29 points between the v1 and v2 benchmarks), so changes are compared on repeats:

    python bench/run_bench.py --tools autocover --subjects slugify dateutil_relativedelta \\
        boltons_iterutils --repeat 3 --out-dir bench/results/repeats --runs-per-day 9
    python bench/report_repeats.py bench/results/repeats

The v1 and v2 benchmark runs of the same subjects are shown for reference (v1 from git,
re-scored; v2 from bench/results/runs/).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
V1_COMMIT = "b58b5af"  # v1 results, re-scored on the shared scorer
METRICS = (("line", "Lines"), ("branch", "Branches"), ("mutation", "Mutation"))


def scores(result: dict) -> dict[str, float]:
    """Line / branch / mutation %, preferring the independent re-score when present."""
    cov = result.get("coverage_rescored") or result["final"]
    mut = result.get("mutation_rescored") or result["mutation"]
    return {"line": cov["line_pct"], "branch": cov["branch_pct"], "mutation": mut["score_pct"],
            "wall": result.get("wall_s", 0.0), "calls": result.get("llm_calls", 0)}


def v1(subject: str) -> dict[str, float] | None:
    spec = f"{V1_COMMIT}:bench/results/runs/{subject}.autocover.json"
    proc = subprocess.run(["git", "show", spec], capture_output=True, text=True,
                          encoding="utf-8", cwd=ROOT)
    return scores(json.loads(proc.stdout)) if proc.returncode == 0 else None


def v2(subject: str) -> dict[str, float] | None:
    path = ROOT / "bench" / "results" / "runs" / f"{subject}.autocover.json"
    return scores(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else None


def _shown(folder: Path) -> str:
    try:
        return folder.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return folder.as_posix()


def main(folder: Path) -> None:
    runs: dict[str, list[dict[str, float]]] = {}
    for path in sorted(folder.glob("*.autocover.r*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result.get("error"):
            runs.setdefault(result["subject"], []).append(scores(result))
    md = ["# Repeated AutoCover-Lite runs\n",
          f"Generated from `{_shown(folder)}`: mean (min-max) over the "
          "repeats, with the single v1 and v2 benchmark runs for reference.\n",
          "| Subject | Runs | " + " | ".join(label for _, label in METRICS)
          + " | v1 lines / mutation | v2 lines / mutation | Wall time |",
          "|---|---|" + "---|" * len(METRICS) + "---|---|---|"]
    for subject, rows in runs.items():
        cells = []
        for key, _ in METRICS:
            values = [r[key] for r in rows]
            cells.append(f"{mean(values):.1f} ({min(values):g}-{max(values):g})")
        refs = [f"{r['line']:g} / {r['mutation']:g}" if r else "-"
                for r in (v1(subject), v2(subject))]
        wall = f"{mean(r['wall'] for r in rows):.0f}s"
        md.append(f"| {subject} | {len(rows)} | " + " | ".join(cells + refs + [wall]) + " |")
    out = folder / "repeats.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out} ({sum(len(r) for r in runs.values())} runs)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "results" / "repeats")
