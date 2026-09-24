"""Executor agent: runs candidates in the sandbox.

Candidates of a round run *batched*: one sandbox run executes many candidate test files,
and per-test coverage contexts credit every line to the exact test that executed it (the
paper's precise coverage attribution). One container per batch instead of one per
candidate removes most of the sandbox start-up cost.

Safety nets: a batch that times out or crashes is re-run one candidate per sandbox, and a
candidate that fails inside a batch is re-run alone to confirm - so one test's side
effects can never make another look broken.
"""

from __future__ import annotations

import asyncio

from autocover.state import Candidate, RunContext, RunState
from autocover.tools.sandbox import RunRequest, RunResult


async def execute(ctx: RunContext, state: RunState) -> RunState:
    pending = state.get("pending", [])
    with ctx.telemetry.span("executor", "run", round=state.get("round", 0),
                            candidates=len(pending)) as span:
        results = await run_candidates(ctx, pending)
        batched = ctx.config.sandbox.batch and len(pending) > 1
        suspects = [c for c in pending if not results[c.id].passed] if batched else []
        if suspects:  # confirm batch failures in isolation
            confirmed = await asyncio.gather(*(run_candidate(ctx, c) for c in suspects))
            results.update({c.id: r for c, r in zip(suspects, confirmed, strict=True)})
        for cand in pending:
            result = results[cand.id]
            ctx.results[cand.id] = result
            ctx.candidates[cand.id] = cand
            cand.duration_s = result.duration_s
            if result.passed:
                cand.status = "passed"
            else:
                cand.status, cand.reason = "failed", failure_reason(result)
                cand.diagnostics = result.diagnostics(limit=1500)
        span.update(passed=sum(c.status == "passed" for c in pending),
                    confirmed_failures=len(suspects))
    return {"pending": [], "executed": pending}


async def run_candidates(ctx: RunContext, cands: list[Candidate], overrides: dict | None = None,
                         timeout_s: float | None = None) -> dict[str, RunResult]:
    """Run candidates (batched when enabled); returns candidate id -> its own result."""
    if not cands:
        return {}
    if not ctx.config.sandbox.batch or len(cands) == 1:
        runs = await asyncio.gather(*(run_candidate(ctx, c, overrides, timeout_s)
                                      for c in cands))
        return {c.id: r for c, r in zip(cands, runs, strict=True)}
    size = max(1, ctx.config.sandbox.batch_size)
    chunks = [cands[i:i + size] for i in range(0, len(cands), size)]
    parts = await asyncio.gather(*(_run_batch(ctx, chunk, overrides, timeout_s)
                                   for chunk in chunks))
    return {cid: r for part in parts for cid, r in part.items()}


async def _run_batch(ctx: RunContext, chunk: list[Candidate], overrides: dict | None,
                     timeout_s: float | None) -> dict[str, RunResult]:
    files = {candidate_filename(c): c for c in chunk}
    base = timeout_s or ctx.config.sandbox.timeout_s
    result = await ctx.sandbox.arun(RunRequest(
        target=ctx.target, tests={name: c.code for name, c in files.items()},
        overrides=overrides or {}, timeout_s=base * 2, per_test=True,
        label=f"batch-{chunk[0].id}+{len(chunk) - 1}"))
    if result.status != "ok":  # timeout/crash: find the culprit one sandbox at a time
        ctx.telemetry.event("executor", "batch_fallback", status=result.status, size=len(chunk))
        runs = await asyncio.gather(*(run_candidate(ctx, c, overrides, timeout_s)
                                      for c in chunk))
        return {c.id: r for c, r in zip(chunk, runs, strict=True)}
    return {c.id: result.for_file(name) for name, c in files.items()}


async def run_candidate(ctx: RunContext, cand: Candidate, overrides: dict | None = None,
                        timeout_s: float | None = None) -> RunResult:
    return await ctx.sandbox.arun(RunRequest(
        target=ctx.target, tests={candidate_filename(cand): cand.code},
        overrides=overrides or {}, timeout_s=timeout_s, label=cand.id))


def candidate_filename(cand: Candidate) -> str:
    return f"test_{cand.id}_{cand.test_name[5:45]}.py"


def failure_reason(result: RunResult) -> str:
    if result.status == "timeout":
        return "timeout"
    if result.collection_errors:
        return "collection error"
    if result.status == "crashed":
        return "sandbox crash"
    return "test failed" if result.failures else "no test collected"
