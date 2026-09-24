"""The agent graph (LangGraph).

    prepare -> generate -> execute -> plan_next --(uncovered lines, rounds & time left)--> generate
                                              \\-> finalize -> END

Milestone 2 wires Preparer, Generator and Executor with a line/branch coverage gate;
the Validator (rules, mutation testing, scenario judge) and the Fixer join in milestone 3.
"""

from __future__ import annotations

from collections import Counter

from langgraph.graph import END, START, StateGraph

from autocover.agents.executor import execute
from autocover.agents.generator import generate
from autocover.agents.preparer import prepare, prioritize
from autocover.state import RunContext, RunState
from autocover.tools.coverage_runner import CoverageTracker
from autocover.tools.sandbox import RunRequest
from autocover.tools.splicer import list_tests, remove_tests

MAX_SUITE_REPAIRS = 2


def build_graph(ctx: RunContext):
    graph = StateGraph(RunState)

    async def prepare_node(state: RunState) -> RunState:
        return await prepare(ctx, state)

    async def generate_node(state: RunState) -> RunState:
        return await generate(ctx, state)

    async def execute_node(state: RunState) -> RunState:
        return await execute(ctx, state)

    async def plan_next_node(state: RunState) -> RunState:
        functions = [ctx.module.function(q) for q in state.get("scenarios", {})]
        return {"targets": prioritize(ctx, functions)}

    async def finalize_node(state: RunState) -> RunState:
        return await finalize(ctx, state)

    def should_continue(state: RunState) -> str:
        if not state.get("targets"):
            return "finalize"
        if state.get("round", 0) >= ctx.config.run.max_rounds or ctx.time_left() <= 0:
            return "finalize"
        return "generate"

    graph.add_node("prepare", prepare_node)
    graph.add_node("generate", generate_node)
    graph.add_node("execute", execute_node)
    graph.add_node("plan_next", plan_next_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", should_continue, ["generate", "finalize"])
    graph.add_edge("generate", "execute")
    graph.add_edge("execute", "plan_next")
    graph.add_conditional_edges("plan_next", should_continue, ["generate", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile()


async def finalize(ctx: RunContext, state: RunState) -> RunState:
    """Run the merged suite as a whole; drop tests that fail together ("do no harm")."""
    suite = state.get("suite", "")
    removed: list[str] = []
    final_cov = None
    with ctx.telemetry.span("finalize", "suite_check") as span:
        for _ in range(MAX_SUITE_REPAIRS + 1):
            if not list_tests(suite):
                break
            result = await ctx.sandbox.arun(RunRequest(
                target=ctx.target, tests={"test_autocover_suite.py": suite}, label="suite"))
            final_cov = result.coverage
            failing = {t.nodeid.split("::")[1].split("[")[0] for t in result.failures
                       if "::" in t.nodeid}
            if result.passed or not failing:
                break
            removed.extend(sorted(failing))
            suite = remove_tests(suite, failing)
        span.update(tests=len(list_tests(suite)), removed=len(removed))

    tracker = CoverageTracker()
    tracker.universe_lines = set(ctx.tracker.universe_lines)
    tracker.universe_branches = set(ctx.tracker.universe_branches)
    if final_cov is not None:
        tracker.add(final_cov)
    history = state.get("history", [])
    statuses = Counter(c.status for c in history)
    final = {
        "target": ctx.target,
        "test_path": ctx.test_path,
        "rounds": state.get("round", 0),
        "baseline": ctx.baseline,
        "final": tracker.summary(),
        "tests_in_suite": len(list_tests(suite)),
        "candidates": len(history),
        "accepted": statuses.get("accepted", 0),
        "failed": statuses.get("failed", 0),
        "rejected_no_gain": statuses.get("rejected", 0),
        "removed_in_suite_check": removed,
        "scenarios_total": sum(len(s) for s in state.get("scenarios", {}).values()),
        "scenarios_covered": len(ctx.covered_scenarios),
        "llm": _llm_usage(ctx),
        "suite": suite,
    }
    return {"suite": suite, "final": final}


def _llm_usage(ctx: RunContext) -> dict:
    calls: Counter = Counter()
    tokens: Counter = Counter()
    for e in ctx.telemetry.events:
        if e.get("stage") == "llm" and e.get("event") == "completion" and not e.get("cached"):
            calls[e["model"]] += 1
            tokens[e["model"]] += e.get("prompt_tokens", 0) + e.get("completion_tokens", 0)
    return {m: {"calls": calls[m], "tokens": tokens[m]} for m in calls}
