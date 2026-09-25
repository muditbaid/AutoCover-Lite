"""Run the benchmark: AutoCover-Lite vs the single-prompt baseline on every subject.

Each (subject, tool) result is written to bench/results/runs/<subject>.<tool>.json as soon
as it finishes, so an interrupted benchmark resumes where it stopped (use --force to redo).
AutoCover-Lite runs once per subject with the full time budget; coverage at shorter budgets
is read off its coverage-over-time curve, as in the paper's Figure 2.

    python bench/prepare_subjects.py
    python bench/run_bench.py --budget-min 15
    python bench/report_bench.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from autocover.config import load_config  # noqa: E402
from autocover.run import run_autocover  # noqa: E402
from autocover.runtime import build_runtime, load_dotenv  # noqa: E402
from autocover.tools.sandbox import make_sandbox  # noqa: E402
from bench.baselines.single_prompt import run_baseline  # noqa: E402
from bench.prepare_subjects import OUT as SUBJECTS_DIR  # noqa: E402
from bench.prepare_subjects import load_subjects, prepare  # noqa: E402

RESULTS = ROOT / "bench" / "results" / "runs"


def bench_config(args):
    cfg = load_config(ROOT / "config.yaml")
    cfg.run.budget_min = args.budget_min
    cfg.run.max_rounds = args.max_rounds
    cfg.run.max_llm_calls = args.max_llm_calls
    cfg.telemetry.jsonl_path = str(ROOT / ".autocover" / "bench_telemetry.jsonl")
    cfg.telemetry.runs_dir = str(ROOT / ".autocover" / "bench_runs")
    cfg.llm.cache.path = str(ROOT / ".autocover" / "llm_cache.sqlite")
    cfg.llm.usage_path = str(ROOT / ".autocover" / "usage.sqlite")
    return cfg


def coverage_curve(events: list[dict], t0: float) -> list[list[float]]:
    """(seconds since start, line %, branch %) after every validation pass."""
    curve = [[0.0, 0.0, 0.0]]
    for e in events:
        if e.get("stage") == "validator" and e.get("event") == "round" and "line_pct" in e:
            curve.append([round(e["ts"] - t0, 1), e["line_pct"], e.get("branch_pct", 0.0)])
    return curve


async def bench_autocover(subject: dict, args) -> dict:
    cfg = bench_config(args)
    # A fresh LLM cache per run: replies cached by earlier runs would make AutoCover-Lite
    # look faster than it is and let it spend the time budget on extra rounds. Caching
    # within the run (retries, repeated prompts) still works.
    fresh = ROOT / ".autocover" / "bench_cache" / f"{subject['name']}-{int(time.time())}.sqlite"
    cfg.llm.cache.path = str(fresh)
    runtime = build_runtime(cfg)
    repo = SUBJECTS_DIR / subject["name"]
    t0 = time.time()
    try:
        summary = await run_autocover(runtime, repo, subject["target"], write=False)
    finally:
        runtime.close()
    events = [e for e in runtime.telemetry.events if e.get("ts", 0) >= t0]
    llm = [e for e in events if e.get("stage") == "llm" and e.get("event") == "completion"
           and not e.get("cached")]
    summary.update({
        "curve": coverage_curve(events, t0),
        "llm_calls": len(llm),
        "llm_tokens": sum(e["prompt_tokens"] + e["completion_tokens"] for e in llm),
        "models": sorted({e["model"] for e in llm}),
        "wall_s": round(time.time() - t0, 1),
    })
    return summary


async def bench_baseline(subject: dict, args) -> dict:
    cfg = bench_config(args)
    runtime = build_runtime(cfg)
    repo = SUBJECTS_DIR / subject["name"]
    try:
        sandbox = make_sandbox(repo, cfg.sandbox, runtime.telemetry)
        sandbox.prepare()
        return await run_baseline(runtime, sandbox, repo, subject["target"],
                                  samples=args.baseline_samples)
    finally:
        runtime.close()


def _engine_down(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(s in text for s in ("SandboxUnavailable", "DockerException",
                                   "All pipe instances are busy"))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--tools", nargs="*", default=["baseline", "autocover"])
    parser.add_argument("--budget-min", type=float, default=15)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--max-llm-calls", type=int, default=200)
    parser.add_argument("--baseline-samples", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    RESULTS.mkdir(parents=True, exist_ok=True)
    for subject in load_subjects(args.subjects):
        if not (SUBJECTS_DIR / subject["name"] / subject["target"]).exists():
            prepare(subject)
        for tool in args.tools:
            out = RESULTS / f"{subject['name']}.{tool}.json"
            if out.exists() and not args.force:
                print(f"[bench] {subject['name']} / {tool}: cached", flush=True)
                continue
            print(f"[bench] {subject['name']} / {tool} ...", flush=True)
            start = time.time()
            try:
                result = await (bench_autocover if tool == "autocover" else bench_baseline)(
                    subject, args)
            except Exception as exc:  # noqa: BLE001 - record and move on
                if _engine_down(exc):
                    # Infrastructure, not a result: don't record it, stop the whole run.
                    print(f"[bench] ABORT: sandbox engine unavailable ({exc}); results so "
                          f"far are kept, re-run to resume", flush=True)
                    return
                result = {"error": f"{type(exc).__name__}: {str(exc)[:500]}"}
            result.update({"subject": subject["name"], "level": subject["level"],
                           "tool": tool, "target": subject["target"],
                           "budget_min": args.budget_min,
                           "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
            out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
            line = result.get("line_pct", (result.get("final") or {}).get("line_pct"))
            mut = (result.get("mutation") or {}).get("score_pct")
            print(f"[bench]   done in {time.time() - start:.0f}s: lines {line}% mutation "
                  f"{mut}% {result.get('error') or ''}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
