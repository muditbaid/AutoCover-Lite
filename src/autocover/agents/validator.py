"""Validator agent: the quality gate between execution and the suite (paper section 3.5).

For every executed candidate:

1. Failed -> the Fixer (with the pytest diagnostics), or frozen after `max_fix_attempts`.
2. Rule gate -> deterministic best-practice checks (`rules/best_practices.yaml`); an
   error-severity violation sends the test to the Fixer with the rule's fix text.
3. Mutation gate -> the test runs against mutants on the lines it executes. A test that
   kills none of them is a weak oracle and goes to the Fixer with the surviving mutants.
4. Acceptance -> greedy by *new signal*: new lines/branches covered, or mutants killed that
   no accepted test kills yet. A test with neither can still be accepted if an LLM judge
   confirms it covers a scenario that no accepted test covers yet.
"""

from __future__ import annotations

import asyncio

from autocover.agents.executor import run_candidate
from autocover.llm.parsing import extract_json
from autocover.llm.prompts import judge_messages
from autocover.llm.router import AllModelsFailed
from autocover.state import Candidate, RunContext, RunState
from autocover.tools.mutator import Mutant, generate_mutants
from autocover.tools.rules import check_test_source, describe
from autocover.tools.splicer import remove_tests, splice_tests

MAX_FEEDBACK_CHARS = 2500
MAX_SURVIVORS_SHOWN = 6


async def validate(ctx: RunContext, state: RunState) -> RunState:
    executed = state.get("executed", [])
    to_fix: list[Candidate] = []
    suite = state.get("suite", "")
    with ctx.telemetry.span("validator", "round", round=state.get("round", 0),
                            candidates=len(executed)) as span:
        clean: list[Candidate] = []
        for cand in executed:
            if cand.status == "failed":
                _route_to_fixer(ctx, cand, cand.diagnostics, to_fix)
                continue
            violations = check_test_source(cand.code, ctx.module.module_name)
            cand.violations = [v.rule for v in violations]
            errors = [v for v in violations if v.is_error]
            if errors:
                cand.reason = "rule: " + ", ".join(dict.fromkeys(v.rule for v in errors))
                _route_to_fixer(ctx, cand, "The test breaks these rules:\n" + describe(errors),
                                to_fix)
            else:
                clean.append(cand)

        weak: list[Candidate] = []
        if ctx.config.mutation.enabled:
            await asyncio.gather(*(_measure_kills(ctx, c) for c in clean))
            weak = [c for c in clean if c.mutants_run and not c.killed]
            for cand in weak:
                cand.reason = f"weak oracle: kills 0 of {cand.mutants_run} mutants"

        suite = await _accept(ctx, clean, suite)
        # Weak oracles: a weak test that adds coverage stays in the suite (its survivors
        # may be equivalent mutants), and the Fixer is asked for a stronger replacement;
        # one that was rejected is sent to the Fixer like any other rejected test.
        for cand in weak:
            if cand.attempt >= ctx.config.run.max_fix_attempts:
                continue
            if cand.status == "accepted":
                cand.diagnostics = _survivors_text(ctx, cand)
                to_fix.append(cand)
            elif cand.status == "rejected":
                _route_to_fixer(ctx, cand, _survivors_text(ctx, cand), to_fix)
        for cand in executed:
            ctx.telemetry.event(
                "validator", "candidate", id=cand.id, function=cand.function,
                test=cand.test_name, attempt=cand.attempt, status=cand.status,
                reason=cand.reason, accepted_by=cand.accepted_by, model=cand.model,
                violations=",".join(cand.violations), mutants_run=cand.mutants_run,
                killed=len(cand.killed), new_lines=len(cand.new_lines))
        span.update(accepted=sum(c.status == "accepted" for c in executed), to_fix=len(to_fix),
                    line_pct=round(ctx.tracker.line_pct, 1),
                    killed_mutants=len(ctx.killed_mutants))
    return {"executed": [], "to_fix": to_fix, "suite": suite,
            "history": [*state.get("history", []), *executed],
            "feedback": _feedback(executed)}


# -- mutation ----------------------------------------------------------------------------


def mutant_pool(ctx: RunContext) -> list[Mutant]:
    if ctx.mutant_pool is None:
        cfg = ctx.config.mutation
        ctx.mutant_pool = generate_mutants(ctx.module.source,
                                           max_per_function=cfg.max_mutants_per_function,
                                           seed=cfg.seed)
    return ctx.mutant_pool


async def _measure_kills(ctx: RunContext, cand: Candidate) -> None:
    executed = ctx.results[cand.id].coverage.executed_lines
    applicable = [m for m in mutant_pool(ctx) if m.lineno in executed]
    applicable.sort(key=lambda m: m.function != cand.function)  # own function first
    applicable = applicable[: ctx.config.mutation.max_mutants_per_candidate]
    runs = await asyncio.gather(*(
        run_candidate(ctx, cand, overrides={ctx.target: m.source},
                      timeout_s=ctx.config.mutation.timeout_s) for m in applicable))
    cand.mutants_run = len(applicable)
    cand.killed = [m.id for m, r in zip(applicable, runs, strict=True) if not r.passed]
    ctx.survivors[cand.id] = [m for m, r in zip(applicable, runs, strict=True) if r.passed]


def _survivors_text(ctx: RunContext, cand: Candidate) -> str:
    lines = ctx.module.source.splitlines()
    survivors = ctx.survivors.get(cand.id, [])
    own = [m for m in survivors if m.function == cand.function]
    shown = (own or survivors)[:MAX_SURVIVORS_SHOWN]
    listed = "\n".join(f"- line {m.lineno}: `{lines[m.lineno - 1].strip()}` with {m.description}"
                       for m in shown)
    return ("The test still passes when the code under test is deliberately broken in each "
            "of these ways, so its assertions are too weak. Strengthen them (exact literal "
            "expected values, boundary inputs) so that the test would fail on these bugs:\n"
            + listed + "\n\nStay within this test's scenario: do not add assertions about "
            "unrelated behaviour, and never inspect signatures or other implementation "
            "details. If a bug above cannot be caught within this scenario, leave it.")


# -- acceptance --------------------------------------------------------------------------


async def _accept(ctx: RunContext, clean: list[Candidate], suite: str) -> str:
    def gain(cand: Candidate):
        cov = ctx.tracker.gain(ctx.results[cand.id].coverage)
        return cov, [k for k in cand.killed if k not in ctx.killed_mutants]

    def score(cand: Candidate) -> int:
        cov, kills = gain(cand)
        return 2 * len(cov.lines) + len(cov.branches) + 2 * len(kills)

    remaining = list(clean)
    while remaining:
        best = max(remaining, key=score)
        if score(best) == 0:
            break
        cov, kills = gain(best)
        best.new_lines, best.new_branches = sorted(cov.lines), sorted(cov.branches)
        best.new_kills = kills
        suite = _accept_one(ctx, best, suite, "coverage" if cov else "mutants")
        remaining.remove(best)

    for cand in remaining:
        if await _judge_accepts(ctx, cand):
            suite = _accept_one(ctx, cand, suite, "scenario")
        else:
            cand.status, cand.reason = "rejected", "no new coverage, mutant kills or scenario"
    return suite


def _accept_one(ctx: RunContext, cand: Candidate, suite: str, by: str) -> str:
    if cand.replaces:  # a stronger version of an accepted weak test: swap it in
        suite = remove_tests(suite, {cand.replaces})
        for other in ctx.candidates.values():
            if (other.status == "accepted" and other.test_name == cand.replaces
                    and other.function == cand.function):
                other.status, other.reason = "superseded", f"replaced by {cand.id}"
    ctx.tracker.add(ctx.results[cand.id].coverage)
    ctx.killed_mutants |= set(cand.killed)
    if cand.scenario_id:
        ctx.covered_scenarios.add((cand.function, cand.scenario_id))
    cand.status, cand.accepted_by = "accepted", by
    if not cand.reason.startswith("weak oracle"):
        cand.reason = ""
    return splice_tests(suite, cand.code).source


async def _judge_accepts(ctx: RunContext, cand: Candidate) -> bool:
    if not (ctx.config.run.judge_scenarios and cand.scenario_id) or \
            (cand.function, cand.scenario_id) in ctx.covered_scenarios:
        return False
    scenario = next((s for s in ctx.scenarios.get(cand.function, []) if s.id == cand.scenario_id),
                    None)
    if scenario is None:
        return False
    fn = ctx.module.function(cand.function)
    try:
        resp = await ctx.router.complete("judge", judge_messages(fn, scenario, cand.code),
                                         json_mode=True, max_tokens=1024)
        verdict = extract_json(resp.text)
    except (AllModelsFailed, ValueError) as exc:
        ctx.telemetry.event("validator", "judge_failed", id=cand.id, error=str(exc)[:200])
        return False
    verdict = verdict if isinstance(verdict, dict) else {}
    covers = verdict.get("covers") is True
    ctx.telemetry.event("validator", "judge", id=cand.id, scenario=scenario.id, covers=covers,
                        reason=str(verdict.get("reason", ""))[:200])
    return covers


# -- routing -----------------------------------------------------------------------------


def _route_to_fixer(ctx: RunContext, cand: Candidate, problem: str,
                    to_fix: list[Candidate]) -> None:
    cand.diagnostics = problem
    if cand.attempt < ctx.config.run.max_fix_attempts:
        cand.status = "needs_fix"
        to_fix.append(cand)
    else:
        cand.status = "frozen"
        cand.reason = (cand.reason or "failed") + f" (frozen after {cand.attempt} fixes)"


def _feedback(candidates: list[Candidate]) -> dict[str, str]:
    """Per function, what went wrong for tests that are out of repair attempts."""
    feedback: dict[str, str] = {}
    for cand in candidates:
        if cand.status != "frozen":
            continue
        tail = "\n".join("    " + line for line in cand.diagnostics.splitlines()[-10:]
                         if line.strip())
        entry = f"- {cand.test_name}: {cand.reason}\n{tail}\n"
        text = feedback.get(cand.function, "")
        if len(text) + len(entry) <= MAX_FEEDBACK_CHARS:
            feedback[cand.function] = text + entry
    return feedback
