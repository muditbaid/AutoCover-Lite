"""The agent graph (LangGraph).

    prepare -> generate -> execute -> validate --(tests to repair)--> fix -> execute -> ...
                                              \\-> plan_next --(gaps left)--> generate
                                                           \\-> finalize -> END

* validate: rule gate, mutation gate, greedy acceptance by new signal, scenario judge.
* fix: repairs rejected/failed tests; repaired tests are re-executed and re-validated.
* plan_next: functions with uncovered lines, then functions with untested scenarios.
* finalize: lint the merged suite, drop tests that fail in combination, measure the
  suite's mutation score, summarise.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from collections import Counter

from langgraph.graph import END, START, StateGraph

from autocover.agents.executor import execute
from autocover.agents.fixer import fix
from autocover.agents.generator import generate
from autocover.agents.preparer import plan_targets, prepare
from autocover.agents.validator import mutant_pool, validate
from autocover.state import RunContext, RunState
from autocover.tools.coverage_runner import CoverageTracker
from autocover.tools.sandbox import RunRequest
from autocover.tools.splicer import list_tests, remove_tests

MAX_SUITE_REPAIRS = 2
FINALIZE_RESERVE_S = 45  # suite check, flaky rerun and mutation score (cap: 10% of budget)
SUITE_FILE = "test_autocover_suite.py"


def build_graph(ctx: RunContext):
    graph = StateGraph(RunState)

    async def prepare_node(state: RunState) -> RunState:
        return await prepare(ctx, state)

    async def generate_node(state: RunState) -> RunState:
        return await generate(ctx, state)

    async def execute_node(state: RunState) -> RunState:
        return await execute(ctx, state)

    async def validate_node(state: RunState) -> RunState:
        return await validate(ctx, state)

    async def fix_node(state: RunState) -> RunState:
        return await fix(ctx, state)

    async def plan_next_node(state: RunState) -> RunState:
        if ctx.round_started:
            ctx.last_round_s = time.monotonic() - ctx.round_started
        return {"targets": plan_targets(ctx, state.get("scenarios", {}))}

    async def finalize_node(state: RunState) -> RunState:
        return await finalize(ctx, state)

    def after_validate(state: RunState) -> str:
        if state.get("to_fix") and ctx.time_left() > 0:
            return "fix"  # fix() itself freezes everything when the LLM budget is gone
        return "plan_next"

    def should_generate(state: RunState) -> str:
        return "generate" if stop_reason(ctx, state) is None else "finalize"

    def after_fix(state: RunState) -> str:
        return "execute" if state.get("pending") else "plan_next"

    for name, node in [("prepare", prepare_node), ("generate", generate_node),
                       ("execute", execute_node), ("validate", validate_node),
                       ("fix", fix_node), ("plan_next", plan_next_node),
                       ("finalize", finalize_node)]:
        graph.add_node(name, node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", should_generate, ["generate", "finalize"])
    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "validate")
    graph.add_conditional_edges("validate", after_validate, ["fix", "plan_next"])
    graph.add_conditional_edges("fix", after_fix, ["execute", "plan_next"])
    graph.add_conditional_edges("plan_next", should_generate, ["generate", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


def stop_reason(ctx: RunContext, state: RunState) -> str | None:
    """Why the run should stop generating (None = keep going)."""
    if not state.get("targets"):
        return "no gaps left"
    if state.get("round", 0) >= ctx.config.run.max_rounds:
        return "max rounds"
    # Start another round only if one more round (as long as the last one) and the final
    # suite checks still fit: the deadline is otherwise only seen between rounds, and a
    # round on a large module can take minutes.
    reserve = min(FINALIZE_RESERVE_S, ctx.config.run.budget_min * 60 * 0.1)
    if ctx.time_left() <= ctx.last_round_s + reserve:
        return "time budget"
    if not ctx.budget_left():
        return "LLM budget"
    return None


def _failing_names(result) -> set[str]:
    return {t.nodeid.split("::")[1].split("[")[0] for t in result.failures if "::" in t.nodeid}


async def finalize(ctx: RunContext, state: RunState) -> RunState:
    """Lint, run the merged suite as a whole ("do no harm"), score it, summarise."""
    suite = lint_fix(state.get("suite", ""))
    removed: list[str] = []
    flaky: list[str] = []
    final_cov = None
    with ctx.telemetry.span("finalize", "suite_check") as span:
        for _ in range(MAX_SUITE_REPAIRS + 1):
            if not list_tests(suite):
                break
            result = await ctx.sandbox.arun(RunRequest(
                target=ctx.target, tests={SUITE_FILE: suite}, label="suite"))
            final_cov = result.coverage
            failing = _failing_names(result)
            if failing:
                removed.extend(sorted(failing))
                suite = remove_tests(suite, failing)
                continue
            if not result.passed:
                break  # crash/timeout without a failing test to blame: keep the suite
            # Flakiness defence: the suite passed once; it must also pass again.
            reruns = await asyncio.gather(*(ctx.sandbox.arun(RunRequest(
                target=ctx.target, tests={SUITE_FILE: suite}, label=f"rerun-{i}"))
                for i in range(ctx.config.run.flaky_reruns)))
            unstable = set().union(*(_failing_names(r) for r in reruns)) if reruns else set()
            if not unstable:
                break
            flaky.extend(sorted(unstable))
            suite = remove_tests(suite, unstable)
        span.update(tests=len(list_tests(suite)), removed=len(removed), flaky=len(flaky))

    history = state.get("history", [])
    mutation = await suite_mutation_score(ctx, suite, history) \
        if ctx.config.mutation.enabled and ctx.config.mutation.final_score else None

    tracker = CoverageTracker()
    tracker.universe_lines = set(ctx.tracker.universe_lines)
    tracker.universe_branches = set(ctx.tracker.universe_branches)
    if final_cov is not None:
        tracker.add(final_cov)
    statuses = Counter(c.status for c in history)
    accepted = [c for c in history if c.status == "accepted"]
    violations = Counter(v for c in history for v in c.violations)
    final = {
        "target": ctx.target,
        "test_path": ctx.test_path,
        "rounds": state.get("round", 0),
        "baseline": ctx.baseline,
        "final": tracker.summary(),
        "mutation": mutation,
        "tests_in_suite": len(list_tests(suite)),
        "candidates": len(history),
        "accepted": len(accepted),
        "accepted_by": dict(Counter(c.accepted_by for c in accepted)),
        "accepted_after_fix": sum(c.attempt > 0 for c in accepted),
        "sent_to_fixer": sum(c.parent_id is not None for c in history),
        "rejected_no_signal": statuses.get("rejected", 0),
        "frozen": statuses.get("frozen", 0),
        "superseded": statuses.get("superseded", 0),
        "weak_kept": sum(c.reason.startswith("weak oracle") for c in accepted),
        "failed": sum(1 for c in history if c.reason in
                      ("test failed", "collection error", "timeout", "no test collected")),
        "rule_violations": dict(violations),
        "removed_in_suite_check": removed,
        "removed_flaky": flaky,
        "stopped_by": stop_reason(ctx, state) or "done",
        "llm_budget": {"calls": ctx.llm_used()[0], "tokens": ctx.llm_used()[1],
                       "max_calls": ctx.config.run.max_llm_calls,
                       "max_tokens": ctx.config.run.max_llm_tokens},
        "scenarios_total": sum(len(s) for s in state.get("scenarios", {}).values()),
        "scenarios_covered": len(ctx.covered_scenarios),
        "llm": _llm_usage(ctx),
        "suite": suite,
    }
    return {"suite": suite, "final": final}


async def suite_mutation_score(ctx: RunContext, suite: str, history: list) -> dict | None:
    """Kill rate of the final suite over the module's (capped) mutant pool.

    Mutants already killed by a test that is still in the suite are known kills (the
    Validator ran them); only the rest are executed against the whole suite.
    """
    names = set(list_tests(suite))
    if not names:
        return None
    pool = mutant_pool(ctx)
    known = {k for c in history if c.status == "accepted" and c.test_name in names
             for k in c.killed}
    unknown = [m for m in pool if m.id not in known]
    with ctx.telemetry.span("finalize", "mutation_score", mutants=len(pool),
                            reused=len(pool) - len(unknown)) as span:
        runs = await asyncio.gather(*(ctx.sandbox.arun(RunRequest(
            target=ctx.target, tests={SUITE_FILE: suite}, overrides={ctx.target: m.source},
            timeout_s=ctx.config.mutation.timeout_s, label=f"score-{m.id}")) for m in unknown))
        survivors = [m for m, r in zip(unknown, runs, strict=True) if r.passed]
        killed = len(pool) - len(survivors)
        score = round(100 * killed / len(pool), 1) if pool else 0.0
        span.update(killed=killed, score=score)
    return {"killed": killed, "total": len(pool), "score_pct": score,
            "reused_kills": len(pool) - len(unknown),
            "survivors": [f"line {m.lineno} ({m.function}): {m.description}"
                          for m in survivors][:20]}


def lint_fix(source: str) -> str:
    """Remove unused imports and sort imports with ruff, when it is installed."""
    if not source.strip():
        return source
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--select", "F401,I", "--fix",
             "--quiet", "--exit-zero", "--stdin-filename", "test_autocover.py", "-"],
            input=source, capture_output=True, encoding="utf-8", errors="replace", timeout=30)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return source  # linting is cosmetic: never let it fail a run
    fixed = proc.stdout
    return fixed if proc.returncode == 0 and fixed.strip() else source


def _llm_usage(ctx: RunContext) -> dict:
    calls: Counter = Counter()
    tokens: Counter = Counter()
    roles: dict[str, set[str]] = {}
    for e in ctx.telemetry.events:
        if e.get("stage") == "llm" and e.get("event") == "completion" and not e.get("cached"):
            calls[e["model"]] += 1
            tokens[e["model"]] += e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
            roles.setdefault(e["model"], set()).add(e.get("role", ""))
    return {m: {"calls": calls[m], "tokens": tokens[m], "roles": sorted(roles[m])}
            for m in calls}
