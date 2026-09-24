"""Entry point for one AutoCover run on one module."""

from __future__ import annotations

import json
import time
from pathlib import Path, PurePosixPath

from autocover.graph import build_graph
from autocover.runtime import Runtime
from autocover.state import RunContext
from autocover.tools.context import build_module_context
from autocover.tools.sandbox import Sandbox, make_sandbox


def default_test_path(target: str) -> str:
    return str(PurePosixPath("tests") / f"test_{Path(target).stem}_autocover.py")


async def run_autocover(
    runtime: Runtime,
    repo: str | Path,
    target: str,
    *,
    test_path: str | None = None,
    functions: list[str] | None = None,
    write: bool = True,
    sandbox: Sandbox | None = None,
) -> dict:
    """Generate tests for `target` (repo-relative module path); returns the run summary."""
    config = runtime.config
    repo = Path(repo).resolve()
    target = Path(target).as_posix()
    test_path = test_path or default_test_path(target)
    box = sandbox or make_sandbox(repo, config.sandbox, runtime.telemetry)
    box.prepare()
    ctx = RunContext(
        config=config, repo=repo, target=target, test_path=test_path,
        router=runtime.router, sandbox=box, telemetry=runtime.telemetry,
        module=build_module_context(repo, target, include_private=config.run.include_private),
        deadline=time.monotonic() + config.run.budget_min * 60, functions=functions,
        telemetry_start=len(runtime.telemetry.events),
    )
    start = time.monotonic()
    with runtime.telemetry.span("run", "autocover", target=target) as span:
        state = await build_graph(ctx).ainvoke({}, {"recursion_limit": 100})
        final = state["final"]
        final["duration_s"] = round(time.monotonic() - start, 1)
        span.update(line_pct=final["final"]["line_pct"], accepted=final["accepted"])
    if write and final["tests_in_suite"]:
        out = repo / test_path
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(final["suite"], encoding="utf-8", newline="\n")
        final["written"] = str(out)
    final["run_id"] = runtime.telemetry.run_id
    if config.telemetry.runs_dir:
        runs = Path(config.telemetry.runs_dir)
        runs.mkdir(parents=True, exist_ok=True)
        (runs / f"{runtime.telemetry.run_id}.json").write_text(
            json.dumps(final, indent=2, default=str), encoding="utf-8")
    return final
