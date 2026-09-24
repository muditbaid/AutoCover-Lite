"""Preparer agent: decides *what* to test and *why*.

1. Initial coverage check - runs any existing test file plus an import probe in the
   sandbox, so the tracker knows which lines are measurable and which are already covered
   (module-level `def` lines run on import and must not count as "new" later).
2. Scenario discovery - one LLM call per function returns happy / edge / error scenarios.
3. Target map - functions ordered by how many of their lines are still uncovered.
4. Scaffolding - the output test file starts from the existing one, if any.
"""

from __future__ import annotations

import asyncio
import re

from autocover.llm.parsing import extract_json
from autocover.llm.prompts import planner_messages
from autocover.llm.router import AllModelsFailed
from autocover.state import RunContext, RunState, Scenario
from autocover.tools.context import FunctionInfo
from autocover.tools.sandbox import RunRequest


async def prepare(ctx: RunContext, state: RunState) -> RunState:
    with ctx.telemetry.span("preparer", "prepare", target=ctx.target) as span:
        suite = _existing_suite(ctx)
        await _baseline(ctx, suite)
        functions = _functions(ctx)
        results = await asyncio.gather(*(_scenarios_for(ctx, fn) for fn in functions))
        scenarios = {fn.qualname: sc for fn, sc in zip(functions, results, strict=True)}
        ctx.scenarios = scenarios
        targets = plan_targets(ctx, scenarios)
        span.update(functions=len(functions),
                    scenarios=sum(len(s) for s in scenarios.values()),
                    baseline_line_pct=round(ctx.tracker.line_pct, 1))
    return {"round": 0, "scenarios": scenarios, "targets": targets, "suite": suite,
            "history": [], "pending": [], "feedback": {}}


def _functions(ctx: RunContext) -> list[FunctionInfo]:
    fns = ctx.module.functions
    if ctx.functions:
        wanted = set(ctx.functions)
        fns = [f for f in fns if f.qualname in wanted]
    return fns


def _existing_suite(ctx: RunContext) -> str:
    path = ctx.repo / ctx.test_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


async def _baseline(ctx: RunContext, suite: str) -> None:
    module = ctx.module.module_name
    tests = {"test_autocover_probe.py":
             f"import {module}\n\n\ndef test_autocover_import_probe():\n"
             f"    assert {module} is not None\n"}
    if suite:
        tests["test_autocover_existing.py"] = suite
    result = await ctx.sandbox.arun(RunRequest(target=ctx.target, tests=tests, label="baseline"))
    if result.coverage is None:
        raise RuntimeError("baseline run produced no coverage:\n" + result.diagnostics())
    if not result.passed:
        # Existing tests fail or the module can't be imported: keep what did run.
        ctx.telemetry.event("preparer", "baseline_failures", details=result.diagnostics(800))
    ctx.tracker.add(result.coverage)
    ctx.baseline = ctx.tracker.summary()


async def _scenarios_for(ctx: RunContext, fn: FunctionInfo) -> list[Scenario]:
    limit = ctx.config.run.max_scenarios_per_function
    try:
        resp = await ctx.router.complete(
            "preparer", planner_messages(ctx.module, fn, limit), json_mode=True,
            max_tokens=ctx.config.run.preparer_max_tokens)
        return parse_scenarios(resp.text, fn.qualname, limit)
    except (AllModelsFailed, ValueError) as exc:
        ctx.telemetry.event("preparer", "scenarios_failed", function=fn.qualname,
                            error=str(exc)[:300])
        # Still generate: the writer gets the uncovered lines instead of scenarios.
        return [Scenario(id="basic", function=fn.qualname, kind="happy",
                         description="typical valid input -> expected return value")]


def parse_scenarios(text: str, function: str, limit: int) -> list[Scenario]:
    data = extract_json(text)
    items = data.get("scenarios", []) if isinstance(data, dict) else data
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not item.get("description"):
            continue
        sid = slug(item.get("id") or item["description"])
        if not sid or sid in seen:
            continue
        seen.add(sid)
        kind = str(item.get("kind", "happy")).lower()
        scenarios.append(Scenario(
            id=sid, function=function, description=str(item["description"])[:300],
            kind=kind if kind in ("happy", "edge", "error") else "happy"))
        if len(scenarios) >= limit:
            break
    if not scenarios:
        raise ValueError("no usable scenarios in reply")
    return scenarios


def plan_targets(ctx: RunContext, scenarios: dict[str, list[Scenario]]) -> list[str]:
    """Functions worth another Generator round: uncovered lines first (most first), then
    functions whose lines are covered but that still have scenarios nobody tested."""
    functions = [ctx.module.function(q) for q in scenarios]
    targets = prioritize(ctx, functions)
    if ctx.config.run.judge_scenarios:
        targets += [q for q, items in scenarios.items() if q not in targets
                    and any((q, s.id) not in ctx.covered_scenarios for s in items)]
    return targets


def prioritize(ctx: RunContext, functions: list[FunctionInfo]) -> list[str]:
    """Functions with uncovered lines first, most uncovered first."""
    uncovered = ctx.tracker.uncovered_lines

    def missing(fn: FunctionInfo) -> int:
        return len(uncovered & set(fn.lines))

    ranked = sorted(functions, key=missing, reverse=True)
    return [fn.qualname for fn in ranked if missing(fn) > 0]


def slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    if s and s[0].isdigit():
        s = "s_" + s
    return s[:max_len].rstrip("_")
