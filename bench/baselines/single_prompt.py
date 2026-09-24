"""Single-prompt baseline: ask the same Generator models once for a whole test module.

Same model chain and the same test-writing rules as AutoCover-Lite, but no scenarios,
no per-function generation, no gates, no repairs. Failing tests are then dropped (the
obvious clean-up anyone would do), and the remainder is scored exactly like AutoCover-Lite:
coverage from one sandbox run, mutation score on the identical mutant pool.
"""

from __future__ import annotations

import asyncio
import time

import libcst

from autocover.agents.validator import build_mutant_pool
from autocover.llm.parsing import extract_code
from autocover.llm.prompts import TEST_RULES
from autocover.llm.router import AllModelsFailed
from autocover.runtime import Runtime
from autocover.tools.context import build_module_context
from autocover.tools.sandbox import RunRequest, Sandbox
from autocover.tools.splicer import list_tests, remove_tests

MAX_SOURCE_CHARS = 60_000
SUITE = "test_baseline_suite.py"

SYSTEM = f"""\
You are a senior Python test engineer. Write a thorough pytest test module for the module
below: cover every public function and method, normal cases, boundaries and error paths.

Rules:
{TEST_RULES}

Reply with a single ```python code block containing the complete test module."""


async def run_baseline(runtime: Runtime, sandbox: Sandbox, repo, target: str,
                       samples: int = 3) -> dict:
    """`samples` independent single-prompt attempts; the median one (by mutation score,
    then line coverage) is reported, all are kept. Sample 0 may come from the cache so
    a re-run is cheap; the others are always fresh."""
    module = build_module_context(repo, target)
    runs = [await _one_sample(runtime, sandbox, module, target, use_cache=(i == 0))
            for i in range(samples)]
    ok = [r for r in runs if "error" not in r]
    if not ok:
        return {"error": runs[0]["error"], "samples": runs}
    ranked = sorted(ok, key=lambda r: (r["mutation"]["score_pct"], r["line_pct"]))
    median = dict(ranked[len(ranked) // 2])
    median["samples"] = [{k: v for k, v in r.items() if k != "suite"} for r in runs]
    median["llm_calls"] = len(ok)
    median["llm_tokens"] = sum(r["llm_tokens"] for r in ok)
    median["wall_s"] = round(sum(r["wall_s"] for r in ok), 1)
    return median


async def _one_sample(runtime: Runtime, sandbox: Sandbox, module, target: str,
                      use_cache: bool) -> dict:
    start = time.monotonic()
    source = module.source[:MAX_SOURCE_CHARS]
    user = (f"Module `{module.module_name}` - import from it with "
            f"`from {module.module_name} import ...`:\n```python\n{source}\n```")
    try:
        resp = await runtime.router.complete(
            "generator", [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}], max_tokens=16384,
            use_cache=use_cache)
    except AllModelsFailed as exc:
        return {"error": f"no model answered: {str(exc)[:200]}"}
    code = extract_code(resp.text)
    try:
        generated = list_tests(code)
    except libcst.ParserSyntaxError:
        generated = []

    raw = await sandbox.arun(RunRequest(target=target, tests={SUITE: code}, label="baseline"))
    failing = {t.nodeid.split("::")[1].split("[")[0] for t in raw.failures if "::" in t.nodeid}
    suite = code if not raw.collection_errors else ""
    if failing and suite:
        suite = remove_tests(suite, failing)
    kept = list_tests(suite) if suite else []
    final = await sandbox.arun(RunRequest(target=target, tests={SUITE: suite},
                                          label="baseline-clean")) if kept else None
    cov = final.coverage if final and final.passed else None
    universe = raw.coverage
    mutation = await score_suite(runtime, sandbox, target, module.source, suite) \
        if cov is not None else {"killed": 0, "total": len(build_mutant_pool(
            module.source, runtime.config.mutation)), "score_pct": 0.0}
    return {
        "model": resp.model,
        "tests_generated": len(generated),
        "tests_passing": len(kept),
        "collection_error": bool(raw.collection_errors),
        "pass_rate": round(100 * len(kept) / len(generated), 1) if generated else 0.0,
        "line_pct": _pct(cov.executed_lines if cov else set(),
                         universe.all_lines if universe else set()),
        "branch_pct": _pct(cov.executed_branches if cov else set(),
                           universe.all_branches if universe else set()),
        "mutation": mutation,
        "llm_calls": 1,
        "llm_tokens": resp.prompt_tokens + resp.completion_tokens,
        "latency_s": round(resp.latency_s, 1),
        "wall_s": round(time.monotonic() - start, 1),
        "suite": suite,
    }


async def score_suite(runtime: Runtime, sandbox: Sandbox, target: str, source: str,
                      suite: str) -> dict:
    """Mutation score of `suite` on the same pool AutoCover-Lite uses."""
    cfg = runtime.config.mutation
    pool = build_mutant_pool(source, cfg)
    runs = await asyncio.gather(*(sandbox.arun(RunRequest(
        target=target, tests={SUITE: suite}, overrides={target: m.source},
        timeout_s=cfg.timeout_s, label=f"baseline-score-{m.id}")) for m in pool))
    killed = sum(not r.passed for r in runs)
    return {"killed": killed, "total": len(pool),
            "score_pct": round(100 * killed / len(pool), 1) if pool else 0.0}


def _pct(part, whole) -> float:
    whole = set(whole)
    return round(100 * len(set(part) & whole) / len(whole), 1) if whole else 0.0
