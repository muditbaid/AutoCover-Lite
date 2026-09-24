"""Milestone 4 operations: flaky-test defence, LLM budget, run reports."""

import asyncio
import shutil

from autocover.config import Config, MutationConfig, RunConfig, TelemetryConfig
from autocover.graph import finalize
from autocover.report import build_report
from autocover.run import run_autocover
from autocover.state import RunContext
from autocover.telemetry import Telemetry
from autocover.tools.context import build_module_context
from autocover.tools.sandbox import LocalSandbox, RunResult, TestOutcome
from tests.test_graph_e2e import EXAMPLE, ScriptedLLM, make_runtime

SUITE = ("from ticket_price import ticket_price\n\n\n"
         "def test_a():\n    assert ticket_price(5) == 5\n\n\n"
         "def test_b():\n    assert ticket_price(30) == 10\n")


class ScriptedSandbox:
    """Returns canned results: the first run passes, later runs fail `test_b` (flaky)."""

    def __init__(self):
        self.calls = 0

    async def arun(self, request):
        self.calls += 1
        names = [n for n in ("test_a", "test_b") if f"def {n}(" in request.tests[
            next(iter(request.tests))]]
        flaky_now = self.calls > 1 and "test_b" in names
        tests = [TestOutcome(f"test_autocover_suite.py::{n}",
                             "failed" if (n == "test_b" and flaky_now) else "passed")
                 for n in names]
        return RunResult(status="ok", exit_code=0, tests=tests)


def test_finalize_drops_tests_that_fail_on_a_rerun(tmp_path):
    config = Config(mutation=MutationConfig(enabled=False), run=RunConfig(flaky_reruns=1),
                    telemetry=TelemetryConfig(jsonl_path=None))
    ctx = RunContext(config=config, repo=EXAMPLE, target="ticket_price.py", test_path="t.py",
                     router=None, sandbox=ScriptedSandbox(), telemetry=Telemetry(),
                     module=build_module_context(EXAMPLE, "ticket_price.py"))
    state = asyncio.run(finalize(ctx, {"suite": SUITE, "history": [], "round": 1}))
    final = state["final"]
    assert final["removed_flaky"] == ["test_b"]
    assert "def test_b" not in state["suite"] and "def test_a" in state["suite"]
    assert final["tests_in_suite"] == 1


def test_llm_budget_stops_the_run(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(EXAMPLE, repo)
    llm = ScriptedLLM()
    # 2 preparer calls + 1 generator call exhaust a budget of 3: no fixes, no round 2.
    runtime = make_runtime(tmp_path, llm, max_llm_calls=3)
    runtime.config.mutation.enabled = False
    box = LocalSandbox(repo, runtime.config.sandbox, runtime.telemetry)
    summary = asyncio.run(run_autocover(runtime, repo, "ticket_price.py", sandbox=box,
                                        write=False, functions=["ticket_price"]))
    assert summary["llm_budget"]["calls"] == 3
    assert summary["stopped_by"] == "LLM budget"
    assert llm.calls["fixer"] == [] and summary["frozen"] >= 1


def test_report_covers_all_sections(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(EXAMPLE, repo)
    runtime = make_runtime(tmp_path, ScriptedLLM())
    runtime.config.telemetry.runs_dir = str(tmp_path / "runs")
    box = LocalSandbox(repo, runtime.config.sandbox, runtime.telemetry)
    summary = asyncio.run(run_autocover(runtime, repo, "ticket_price.py", sandbox=box,
                                        write=False))
    assert (tmp_path / "runs" / f"{summary['run_id']}.json").exists()
    text = build_report(runtime.telemetry.events, summary)
    for section in ("time by stage", "sandbox", "llm (role", "candidate funnel",
                    "mutation score", "stopped by"):
        assert section in text, section
    assert "batched" in text and "accepted by:" in text


def test_replacement_that_covers_less_keeps_the_weak_test():
    from autocover.agents.validator import _accept_one
    from autocover.state import Candidate
    from autocover.tools.sandbox import CoverageData

    ctx = RunContext(config=Config(), repo=EXAMPLE, target="ticket_price.py",
                     test_path="t.py", router=None, sandbox=None, telemetry=Telemetry(),
                     module=build_module_context(EXAMPLE, "ticket_price.py"))
    weak = Candidate(id="c1", function="ticket_price", test_name="test_tp", round=1,
                     code="def test_tp():\n    assert True\n", status="accepted")
    ctx.candidates["c1"] = weak
    ctx.results["c1"] = RunResult(status="ok", coverage=CoverageData(
        executed_lines=frozenset({6, 7, 8}), missing_lines=frozenset({9})))
    suite = weak.code

    narrower = Candidate(id="c2", function="ticket_price", test_name="test_tp", round=1,
                         code="def test_tp():\n    assert 1 == 1\n", replaces="test_tp")
    ctx.results["c2"] = RunResult(status="ok", coverage=CoverageData(
        executed_lines=frozenset({6, 7}), missing_lines=frozenset({8, 9})))
    suite = _accept_one(ctx, narrower, suite, "mutants")
    assert weak.status == "accepted" and suite.count("def test_tp") == 2  # both kept

    wider = Candidate(id="c3", function="ticket_price", test_name="test_tp", round=1,
                      code="def test_tp():\n    assert 2 == 2\n", replaces="test_tp")
    ctx.results["c3"] = RunResult(status="ok", coverage=CoverageData(
        executed_lines=frozenset({6, 7, 8, 9}), missing_lines=frozenset()))
    _accept_one(ctx, wider, weak.code, "mutants")
    assert weak.status == "superseded"
