"""Re-score every AutoCover-Lite suite with the baseline's scorer (integrity check).

AutoCover-Lite's own final mutation score reuses kills recorded during validation (each
test was run against its mutants in isolation). This script ignores those and runs the
full, identical mutant pool against each final suite from scratch - exactly how the
baseline is scored - and stores the result as `mutation_rescored` next to the original.

    python bench/rescore.py            # all subjects with an AutoCover-Lite result
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from autocover.runtime import build_runtime, load_dotenv  # noqa: E402
from autocover.tools.context import build_module_context  # noqa: E402
from autocover.tools.sandbox import make_sandbox  # noqa: E402
from bench.baselines.single_prompt import score_suite  # noqa: E402
from bench.prepare_subjects import OUT as SUBJECTS_DIR  # noqa: E402
from bench.run_bench import RESULTS, bench_config  # noqa: E402


class _Args:
    budget_min, max_rounds, max_llm_calls = 15, 10, 200


async def rescore(path: Path) -> None:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("error") or not result.get("suite"):
        print(f"[rescore] {path.name}: skipped (no suite)")
        return
    repo = SUBJECTS_DIR / result["subject"]
    cfg = bench_config(_Args)
    runtime = build_runtime(cfg)
    try:
        sandbox = make_sandbox(repo, cfg.sandbox, runtime.telemetry)
        sandbox.prepare()
        source = build_module_context(repo, result["target"]).source
        rescored = await score_suite(runtime, sandbox, result["target"], source, result["suite"])
    finally:
        runtime.close()
    result["mutation_rescored"] = rescored
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    own = (result.get("mutation") or {}).get("score_pct")
    print(f"[rescore] {result['subject']:<24} own {own}%  rescored {rescored['score_pct']}% "
          f"({rescored['killed']}/{rescored['total']})", flush=True)


async def main() -> None:
    load_dotenv(ROOT / ".env")
    for path in sorted(RESULTS.glob("*.autocover.json")):
        await rescore(path)


if __name__ == "__main__":
    asyncio.run(main())
