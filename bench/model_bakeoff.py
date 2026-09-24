"""Compare Generator models on the same module and the same scenarios.

For each model: one Generator round (max_rounds=1) with that model as the only generator.
The Preparer's scenario replies are cached after the first model, so every model gets the
identical prompts. Writes bench/results/model_bakeoff.{md,json}.

    python bench/model_bakeoff.py                       # all generator/fixer chain models
    python bench/model_bakeoff.py --models a/x b/y      # specific models
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autocover.config import load_config  # noqa: E402
from autocover.run import run_autocover  # noqa: E402
from autocover.runtime import build_runtime, load_dotenv  # noqa: E402


def chain_models(config) -> list[str]:
    seen: list[str] = []
    for role in ("generator", "fixer"):
        for model in config.llm.roles.get(role, []):
            if model not in seen:
                seen.append(model)
    return seen


async def bake(model: str, repo: Path, target: str, config_path: Path) -> dict:
    cfg = load_config(config_path)
    cfg.llm.roles["generator"] = [model]
    cfg.run.max_rounds = 1
    cfg.run.max_fix_attempts = 0  # measure the generator alone: no repairs
    runtime = build_runtime(cfg)
    start = time.monotonic()
    try:
        summary = await run_autocover(runtime, repo, target, write=False)
    except Exception as exc:  # noqa: BLE001 - record and continue with the next model
        return {"model": model, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    finally:
        runtime.close()
    gen = [e for e in runtime.telemetry.events
           if e.get("event") == "completion" and e.get("role") == "generator"]
    cands = summary["candidates"]
    first_try = [e for e in runtime.telemetry.events
                 if e.get("event") == "candidate" and e.get("attempt") == 0]
    passed = sum(e["status"] not in ("needs_fix", "frozen") or "weak" in e.get("reason", "")
                 for e in first_try)
    return {
        "model": model,
        "candidates": cands,
        "pass_rate": round(100 * passed / cands, 1) if cands else 0.0,
        "accepted": summary["accepted"],
        "line_pct": summary["final"]["line_pct"],
        "branch_pct": summary["final"]["branch_pct"],
        "gen_calls": len(gen),
        "gen_latency_s": round(sum(e["latency_s"] for e in gen) / len(gen), 1) if gen else None,
        "gen_tokens": sum(e["prompt_tokens"] + e["completion_tokens"] for e in gen),
        "wall_s": round(time.monotonic() - start, 1),
        "error": None if gen else "no generator reply (quota, error or all fallbacks skipped)",
    }


def to_markdown(rows: list[dict], target: str) -> str:
    head = ("| Model | Candidates | Pass rate | Accepted | Lines | Branches | "
            "Latency/call | Tokens | Notes |\n|---|---|---|---|---|---|---|---|---|\n")
    body = ""
    for r in sorted(rows, key=lambda r: (-(r.get("line_pct") or 0), -(r.get("pass_rate") or 0))):
        if r.get("candidates") is None:
            body += f"| `{r['model']}` | - | - | - | - | - | - | - | {r['error']} |\n"
            continue
        body += (f"| `{r['model']}` | {r['candidates']} | {r['pass_rate']}% | {r['accepted']} | "
                 f"{r['line_pct']}% | {r['branch_pct']}% | {r['gen_latency_s']}s | "
                 f"{r['gen_tokens']} | {r['error'] or ''} |\n")
    return (f"# Generator model bake-off: `{target}`\n\nOne Generator round per model, "
            f"identical cached scenarios. Run {time.strftime('%Y-%m-%d %H:%M')}.\n\n"
            + head + body)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(ROOT / "examples" / "ticket_price"))
    parser.add_argument("--target", default="ticket_price.py")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--out", default=str(ROOT / "bench" / "results" / "model_bakeoff"))
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    models = args.models or chain_models(load_config(args.config))
    rows = []
    for model in models:
        print(f"[bakeoff] {model} ...", flush=True)
        row = await bake(model, Path(args.repo), args.target, Path(args.config))
        print(f"[bakeoff]   {row}", flush=True)
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    out.with_suffix(".md").write_text(to_markdown(rows, args.target), encoding="utf-8")
    print(f"[bakeoff] wrote {out.with_suffix('.md')}")


if __name__ == "__main__":
    asyncio.run(main())
