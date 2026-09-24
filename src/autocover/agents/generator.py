"""Generator agent: proposes tests that should raise line coverage or cover new scenarios.

One LLM call per target function (all of its open scenarios at once - cheaper on free
tiers than one call per scenario); the functions of a round run in parallel. The reply is
split into single-test candidates, so each test is executed, credited with coverage and
accepted or rejected on its own. Candidates identical to ones already tried are dropped.
"""

from __future__ import annotations

import asyncio
import hashlib
import re

import libcst

from autocover.llm.parsing import extract_code
from autocover.llm.prompts import stem_of, writer_messages
from autocover.llm.router import AllModelsFailed
from autocover.state import Candidate, RunContext, RunState, Scenario
from autocover.tools.splicer import split_tests


async def generate(ctx: RunContext, state: RunState) -> RunState:
    round_no = state.get("round", 0) + 1
    if not ctx.budget_left():
        ctx.telemetry.event("generator", "budget_exhausted", round=round_no)
        return {"round": round_no, "pending": []}
    with ctx.telemetry.span("generator", "round", round=round_no) as span:
        targets = state.get("targets", [])[: ctx.config.run.max_functions_per_round]
        batches = await asyncio.gather(
            *(_generate_for(ctx, state, fn, round_no) for fn in targets))
        pending = [c for batch in batches for c in batch]
        span.update(functions=len(batches), candidates=len(pending))
    return {"round": round_no, "pending": pending}


async def _generate_for(ctx: RunContext, state: RunState, qualname: str,
                        round_no: int) -> list[Candidate]:
    fn = ctx.module.function(qualname)
    scenarios = [s for s in state.get("scenarios", {}).get(qualname, [])
                 if (qualname, s.id) not in ctx.covered_scenarios]
    uncovered = uncovered_source(ctx, qualname)
    survivors = survivors_for(ctx, qualname)
    if not scenarios and not uncovered and not survivors:
        return []
    messages = writer_messages(ctx.module, fn, scenarios, uncovered,
                               state.get("feedback", {}).get(qualname, ""), survivors)
    try:
        resp = await ctx.router.complete("generator", messages,
                                         max_tokens=ctx.config.run.generator_max_tokens)
    except AllModelsFailed as exc:
        ctx.telemetry.event("generator", "llm_failed", function=qualname, error=str(exc)[:300])
        return []
    return make_candidates(ctx, qualname, scenarios, extract_code(resp.text),
                           round_no, resp.model)


def make_candidates(ctx: RunContext, qualname: str, scenarios: list[Scenario], code: str,
                    round_no: int, model: str) -> list[Candidate]:
    try:
        parts = split_tests(code)
    except libcst.ParserSyntaxError:
        parts = [(f"test_{stem_of(qualname)}__unparsable", code)]  # fails -> feedback
    ids = {s.id for s in scenarios}
    prefix = f"test_{stem_of(qualname)}__"
    candidates = []
    for name, source in parts:
        digest = hashlib.sha256(normalize_code(source).encode()).hexdigest()
        if digest in ctx.seen_code:
            ctx.telemetry.event("generator", "duplicate_dropped", function=qualname, test=name)
            continue
        ctx.seen_code.add(digest)
        sid = name[len(prefix):] if name.startswith(prefix) else None
        candidates.append(Candidate(
            id=ctx.next_id(), function=qualname, test_name=name,
            scenario_id=sid if sid in ids else None, code=source,
            round=round_no, model=model))
    return candidates


def uncovered_source(ctx: RunContext, qualname: str) -> list[tuple[int, str]]:
    """(line number, source text) of measurable lines the function can reach (its own and
    its private helpers') that no test covers yet; helper lines name their helper."""
    fn = ctx.module.function(qualname)
    missing = sorted(ctx.tracker.uncovered_lines & ctx.module.reach(fn))
    source_lines = ctx.module.source.splitlines()
    out = []
    for n in missing:
        if not 0 < n <= len(source_lines):
            continue
        owner = ctx.module.owner_of(n)
        where = f"  [in {owner}]" if owner and owner != qualname else ""
        out.append((n, source_lines[n - 1].strip() + where))
    return out


def survivors_for(ctx: RunContext, qualname: str, limit: int = 8) -> list[tuple[int, str, str]]:
    """Surviving mutants in the function's reach, as (line, code, change); marks them
    shown so each is put in front of the Generator only once."""
    if not ctx.config.mutation.enabled:
        return []
    reach = ctx.module.reach(ctx.module.function(qualname))
    picked = [m for m in ctx.open_survivors() if m.lineno in reach][:limit]
    ctx.shown_survivors |= {m.id for m in picked}
    lines = ctx.module.source.splitlines()
    return [(m.lineno, lines[m.lineno - 1].strip(), m.description) for m in picked]


def normalize_code(source: str) -> str:
    """Whitespace/comment-insensitive form used for duplicate detection."""
    no_comments = re.sub(r"#[^\n]*", "", source)
    return re.sub(r"\s+", " ", no_comments).strip()
