"""Fixer agent: repairs failing or rejected tests with targeted edits (paper section 3.6).

Input is a candidate the Validator sent back, with the reason: pytest diagnostics, rule
violations (with the registry's fix instructions), or the mutants its assertions let
survive. The repaired test is a new candidate (attempt + 1, parent_id = original) that goes
back through the Executor and the Validator. A repair identical to a version already tried
freezes the test instead of looping; so does running out of attempts. Harmful repairs never
reach the suite: every repaired test is re-validated on its own, and the finalize step
re-checks the merged suite.
"""

from __future__ import annotations

import asyncio
import hashlib

import libcst

from autocover.agents.generator import normalize_code
from autocover.llm.parsing import extract_code
from autocover.llm.prompts import fixer_messages
from autocover.llm.router import AllModelsFailed
from autocover.state import Candidate, RunContext, RunState
from autocover.tools.splicer import split_tests


async def fix(ctx: RunContext, state: RunState) -> RunState:
    to_fix = state.get("to_fix", [])
    with ctx.telemetry.span("fixer", "round", candidates=len(to_fix)) as span:
        repaired = await asyncio.gather(*(_fix_one(ctx, c) for c in to_fix))
        pending = [c for c in repaired if c is not None]
        span.update(repaired=len(pending), frozen=len(to_fix) - len(pending))
    return {"to_fix": [], "pending": pending}


async def _fix_one(ctx: RunContext, cand: Candidate) -> Candidate | None:
    fn = ctx.module.function(cand.function)
    messages = fixer_messages(ctx.module, fn, cand.code, cand.test_name, cand.diagnostics)
    try:
        resp = await ctx.router.complete("fixer", messages,
                                         max_tokens=ctx.config.run.generator_max_tokens)
    except AllModelsFailed as exc:
        return _freeze(ctx, cand, f"fixer unavailable: {str(exc)[:120]}")
    code = extract_code(resp.text)
    try:
        parts = split_tests(code)
    except libcst.ParserSyntaxError:
        parts = [(cand.test_name, code)]  # let the sandbox report the syntax error
    if not parts:
        return _freeze(ctx, cand, "fixer returned no test")
    name, source = next(((n, s) for n, s in parts if n == cand.test_name), parts[0])
    digest = hashlib.sha256(normalize_code(source).encode()).hexdigest()
    if digest in ctx.seen_code:
        return _freeze(ctx, cand, "fixer repeated an already tried version")
    ctx.seen_code.add(digest)
    return Candidate(
        id=ctx.next_id(), function=cand.function, test_name=name,
        scenario_id=cand.scenario_id, code=source, round=cand.round, model=resp.model,
        attempt=cand.attempt + 1, parent_id=cand.id,
        replaces=cand.test_name if cand.status == "accepted" else cand.replaces)


def _freeze(ctx: RunContext, cand: Candidate, why: str) -> None:
    """Stop repairing `cand`. An accepted weak test keeps its place in the suite."""
    if cand.status != "accepted":
        cand.status = "frozen"
    cand.reason = f"{cand.reason}; {why}" if cand.reason else why
    ctx.telemetry.event("fixer", "frozen", id=cand.id, test=cand.test_name, reason=why)
    return None
