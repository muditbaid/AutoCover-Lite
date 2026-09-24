"""Executor agent: runs candidates in the sandbox and keeps the ones that add signal.

* Every pending candidate runs in its own sandbox (in parallel, bounded by
  `sandbox.max_parallel`) and is credited with exactly the coverage it produced.
* Coverage gate (milestone 2): passing candidates are considered biggest-gain first and a
  candidate is accepted only if it executes lines or branches the accepted suite does not
  yet cover. Accepted tests are spliced into the suite with AST-aware edits.
* Failures are summarised per function and fed back to the Generator's next round.
"""

from __future__ import annotations

import asyncio

from autocover.state import Candidate, RunContext, RunState
from autocover.tools.sandbox import RunRequest, RunResult
from autocover.tools.splicer import splice_tests

MAX_FEEDBACK_CHARS = 2500


async def execute(ctx: RunContext, state: RunState) -> RunState:
    pending = state.get("pending", [])
    with ctx.telemetry.span("executor", "round", round=state.get("round", 0),
                            candidates=len(pending)) as span:
        results = await asyncio.gather(*(_run(ctx, c) for c in pending))
        for cand, result in zip(pending, results, strict=True):
            cand.duration_s = result.duration_s
            if result.passed:
                cand.status = "passed"
            else:
                cand.status, cand.reason = "failed", _failure_reason(result)
                cand.diagnostics = result.diagnostics(limit=1500)
        suite = state.get("suite", "")
        accepted = 0
        for cand, result in _by_gain(ctx, pending, results):
            gain = ctx.tracker.gain(result.coverage)
            if not gain:
                cand.status, cand.reason = "rejected", "no new line or branch coverage"
                continue
            spliced = splice_tests(suite, cand.code)
            suite = spliced.source
            ctx.tracker.add(result.coverage)
            cand.status = "accepted"
            cand.new_lines = sorted(gain.lines)
            cand.new_branches = sorted(gain.branches)
            if cand.scenario_id:
                ctx.covered_scenarios.add((cand.function, cand.scenario_id))
            accepted += 1
        for cand in pending:
            ctx.telemetry.event("executor", "candidate", id=cand.id, function=cand.function,
                                test=cand.test_name, status=cand.status, reason=cand.reason,
                                model=cand.model, new_lines=len(cand.new_lines))
        span.update(accepted=accepted, failed=sum(c.status == "failed" for c in pending),
                    line_pct=round(ctx.tracker.line_pct, 1),
                    branch_pct=round(ctx.tracker.branch_pct, 1))
    return {"pending": [], "suite": suite, "history": [*state.get("history", []), *pending],
            "feedback": _feedback(pending)}


async def _run(ctx: RunContext, cand: Candidate) -> RunResult:
    filename = f"test_{cand.id}_{cand.test_name[5:45]}.py"
    return await ctx.sandbox.arun(RunRequest(target=ctx.target, tests={filename: cand.code},
                                             label=cand.id))


def _by_gain(ctx: RunContext, pending: list[Candidate], results: list[RunResult]):
    """Passing (candidate, result) pairs, largest potential gain first (greedy cover)."""
    passing = [(c, r) for c, r in zip(pending, results, strict=True)
               if c.status == "passed" and r.coverage is not None]

    def potential(pair) -> int:
        gain = ctx.tracker.gain(pair[1].coverage)
        return len(gain.lines) * 2 + len(gain.branches)

    return sorted(passing, key=potential, reverse=True)


def _failure_reason(result: RunResult) -> str:
    if result.status == "timeout":
        return "timeout"
    if result.collection_errors:
        return "collection error"
    if result.status == "crashed":
        return "sandbox crash"
    return "test failed" if result.failures else "no test collected"


def _feedback(candidates: list[Candidate]) -> dict[str, str]:
    """Per function, a compact list of what failed this round (for the next prompt)."""
    feedback: dict[str, str] = {}
    for cand in candidates:
        if cand.status != "failed":
            continue
        entry = f"- {cand.test_name}: {cand.reason}\n{_last_lines(cand.diagnostics, 12)}\n"
        text = feedback.get(cand.function, "")
        if len(text) + len(entry) <= MAX_FEEDBACK_CHARS:
            feedback[cand.function] = text + entry
    return feedback


def _last_lines(text: str, n: int) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    return "\n".join("    " + line for line in lines[-n:])
