"""Executor agent: runs candidates in the sandbox.

Every pending candidate (fresh from the Generator or repaired by the Fixer) runs in its own
sandbox, in parallel, bounded by `sandbox.max_parallel`. The run result - per-test
outcomes and exactly the coverage this candidate produced - is kept for the Validator.
"""

from __future__ import annotations

import asyncio

from autocover.state import Candidate, RunContext, RunState
from autocover.tools.sandbox import RunRequest, RunResult


async def execute(ctx: RunContext, state: RunState) -> RunState:
    pending = state.get("pending", [])
    with ctx.telemetry.span("executor", "run", round=state.get("round", 0),
                            candidates=len(pending)) as span:
        results = await asyncio.gather(*(run_candidate(ctx, c) for c in pending))
        for cand, result in zip(pending, results, strict=True):
            ctx.results[cand.id] = result
            ctx.candidates[cand.id] = cand
            cand.duration_s = result.duration_s
            if result.passed:
                cand.status = "passed"
            else:
                cand.status, cand.reason = "failed", failure_reason(result)
                cand.diagnostics = result.diagnostics(limit=1500)
        span.update(passed=sum(c.status == "passed" for c in pending))
    return {"pending": [], "executed": pending}


async def run_candidate(ctx: RunContext, cand: Candidate, overrides: dict | None = None,
                        timeout_s: float | None = None) -> RunResult:
    filename = f"test_{cand.id}_{cand.test_name[5:45]}.py"
    return await ctx.sandbox.arun(RunRequest(
        target=ctx.target, tests={filename: cand.code}, overrides=overrides or {},
        timeout_s=timeout_s, label=cand.id))


def failure_reason(result: RunResult) -> str:
    if result.status == "timeout":
        return "timeout"
    if result.collection_errors:
        return "collection error"
    if result.status == "crashed":
        return "sandbox crash"
    return "test failed" if result.failures else "no test collected"
