"""End-to-end runs of the agent graph with a scripted LLM and the local sandbox.

The script exercises every Validator path: a failing test the Fixer repairs, a computed
oracle caught by the rule gate, a weak oracle caught by mutation testing, a test accepted
only for its mutant kills, one accepted by the scenario judge, and a redundant one.
"""

import asyncio
import json
import re
import shutil
from pathlib import Path

import pytest

from autocover.config import (
    Config,
    LLMConfig,
    MutationConfig,
    RunConfig,
    SandboxConfig,
    TelemetryConfig,
)
from autocover.llm.cache import UsageLedger
from autocover.llm.router import LLMRouter
from autocover.run import run_autocover
from autocover.runtime import Runtime
from autocover.telemetry import Telemetry
from autocover.tools.sandbox import LocalSandbox, RunRequest

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ticket_price"

SCENARIOS = {
    "ticket_price": [
        {"id": "child", "kind": "happy", "description": "age 5 -> 5"},
        {"id": "senior_boundary", "kind": "edge", "description": "age 65 -> 7"},
        {"id": "adult", "kind": "happy", "description": "age 30 -> 10"},
        {"id": "negative_age", "kind": "error", "description": "age -1 -> ValueError"},
        {"id": "child_boundary", "kind": "edge", "description": "age 12 -> 10 (adult)"},
        {"id": "senior_old", "kind": "happy", "description": "age 70 -> 7"},
    ],
    "group_total": [
        {"id": "small_group", "kind": "happy", "description": "2 adults -> 20.0"},
        {"id": "discount", "kind": "happy", "description": "5 adults, 10% off -> 45.0"},
    ],
}

TICKET_TESTS = '''```python
import pytest
from ticket_price import ticket_price


def test_ticket_price__child():
    assert ticket_price(5) == 5


def test_ticket_price__senior_boundary():
    assert ticket_price(65) == 7


def test_ticket_price__adult():
    assert ticket_price(30) == 10


def test_ticket_price__negative_age():
    with pytest.raises(ValueError):
        ticket_price(-1)


def test_ticket_price__child_boundary():
    assert ticket_price(12) == 5


def test_ticket_price__senior_old():
    assert ticket_price(70) == 7


def test_ticket_price__another_child():
    assert ticket_price(3) == 5
```'''

GROUP_COMPUTED = '''```python
from ticket_price import group_total, ticket_price


def test_group_total__small_group():
    ages = [30, 30]
    expected = sum(ticket_price(a) for a in ages)
    assert group_total(ages) == expected
```'''

GROUP_DISCOUNT = '''```python
from ticket_price import group_total


def test_group_total__discount():
    assert group_total([30] * 5, discount=0.1) == 45.0
```'''

FIXES = {
    "test_ticket_price__child_boundary": (
        "from ticket_price import ticket_price\n\n\n"
        "def test_ticket_price__child_boundary():\n    assert ticket_price(12) == 10\n"),
    "test_ticket_price__negative_age": (
        "import pytest\nfrom ticket_price import ticket_price\n\n\n"
        "def test_ticket_price__negative_age():\n"
        "    with pytest.raises(ValueError, match='non-negative'):\n        ticket_price(-1)\n"),
    "test_group_total__small_group": (
        "from ticket_price import group_total\n\n\n"
        "def test_group_total__small_group():\n    assert group_total([30, 30]) == 20.0\n"),
}


class ScriptedLLM:
    def __init__(self):
        self.calls: dict[str, list[str]] = {"planner": [], "writer": [], "fixer": [],
                                            "judge": []}

    async def __call__(self, *, model, messages, **kwargs):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "SCENARIO_PLANNER" in system:
            self.calls["planner"].append(user)
            fn = re.search(r"scenarios for `([\w.]+)`", user).group(1)
            text = json.dumps({"scenarios": SCENARIOS[fn]})
        elif "TEST_FIXER" in system:
            self.calls["fixer"].append(user)
            name = re.search(r"Test `(test_\w+)`", user).group(1)
            text = f"```python\n{FIXES[name]}```"
        elif "SCENARIO_JUDGE" in system:
            self.calls["judge"].append(user)
            text = json.dumps({"covers": True, "reason": "asserts the stated behaviour"})
        else:
            self.calls["writer"].append(user)
            if "test_group_total__" in user:
                text = GROUP_DISCOUNT if "test_group_total__discount" in user and \
                    "test_group_total__small_group" not in user else GROUP_COMPUTED
            else:
                text = TICKET_TESTS
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(EXAMPLE, dest)
    return dest


def make_runtime(tmp_path, llm, **run_overrides):
    roles = {role: [f"fake/{role}"] for role in ("preparer", "generator", "fixer", "judge")}
    config = Config(
        llm=LLMConfig(roles=roles),
        sandbox=SandboxConfig(backend="local", workdir=str(tmp_path / "ws"), max_parallel=6,
                              timeout_s=30),
        # All mutants in the pool, so the kill-based assertions are deterministic.
        mutation=MutationConfig(max_mutants_per_function=50, max_mutants_per_candidate=30,
                                timeout_s=20),
        run=RunConfig(max_rounds=3, **run_overrides),
        telemetry=TelemetryConfig(jsonl_path=None),
    )
    telemetry = Telemetry()
    router = LLMRouter(config.llm, completion_fn=llm, require_keys=False, telemetry=telemetry)
    return Runtime(config=config, telemetry=telemetry, router=router, cache=None,
                   usage=UsageLedger())


def outcomes(runtime) -> dict[str, list[dict]]:
    by_test: dict[str, list[dict]] = {}
    for e in runtime.telemetry.events:
        if e.get("stage") == "validator" and e.get("event") == "candidate":
            by_test.setdefault(e["test"], []).append(e)
    return by_test


def test_quality_loop_end_to_end(tmp_path, repo):
    llm = ScriptedLLM()
    runtime = make_runtime(tmp_path, llm)
    box = LocalSandbox(repo, runtime.config.sandbox, runtime.telemetry)
    summary = asyncio.run(run_autocover(runtime, repo, "ticket_price.py", sandbox=box))
    seen = outcomes(runtime)

    # Wrong expectation -> failed -> repaired -> accepted for mutants no other test kills
    # (`age < 12` -> `age <= 12`), since it adds no new lines.
    boundary = seen["test_ticket_price__child_boundary"]
    assert [e["status"] for e in boundary] == ["needs_fix", "accepted"]
    assert boundary[-1]["accepted_by"] == "mutants"

    # Computed oracle -> rule gate -> repaired with a literal -> accepted.
    small = seen["test_group_total__small_group"]
    assert small[0]["status"] == "needs_fix" and "literal_oracle" in small[0]["violations"]
    assert small[-1]["status"] == "accepted"

    # pytest.raises without checking the message kills no mutant: the weak test is kept
    # for its coverage (only it reaches the `raise`), and a stronger version replaces it.
    neg = seen["test_ticket_price__negative_age"]
    assert neg[0]["status"] == "accepted" and neg[0]["reason"].startswith("weak oracle")
    assert neg[-1]["status"] == "accepted" and neg[-1]["attempt"] == 1

    # No new lines or kills, but an untested scenario confirmed by the judge.
    assert seen["test_ticket_price__senior_old"][-1]["accepted_by"] == "scenario"
    # Redundant and not tied to a scenario -> rejected.
    assert seen["test_ticket_price__another_child"][-1]["status"] == "rejected"

    assert summary["final"]["line_pct"] == 100.0 and summary["final"]["branch_pct"] == 100.0
    assert summary["accepted_after_fix"] == 3
    assert summary["rule_violations"].get("literal_oracle", 0) >= 1
    assert summary["scenarios_covered"] == summary["scenarios_total"] == 8
    # 18/24: the survivors are real gaps of this scripted suite (no age-0 boundary test,
    # no group of 5 without discount) plus near-equivalent mutants; they get reported.
    assert summary["mutation"]["score_pct"] >= 70
    assert "line 6 (ticket_price): < -> <=" in summary["mutation"]["survivors"]
    assert summary["removed_in_suite_check"] == []
    assert len(llm.calls["fixer"]) == 3 and len(llm.calls["judge"]) >= 1

    written = (repo / "tests" / "test_ticket_price_autocover.py").read_text()
    assert "assert ticket_price(12) == 10" in written
    assert "match='non-negative'" in written
    assert written.count("def test_ticket_price__negative") == 1
    assert summary["superseded"] == 1
    assert "expected = sum(" not in written and "another_child" not in written
    check = box.run(RunRequest(target="ticket_price.py", tests={"test_final.py": written}))
    assert check.passed and len(check.tests) == summary["tests_in_suite"]


def test_without_mutation_or_fixer_behaves_like_a_coverage_gate(tmp_path, repo):
    llm = ScriptedLLM()
    runtime = make_runtime(tmp_path, llm, max_fix_attempts=0, judge_scenarios=False)
    runtime.config.mutation.enabled = False
    box = LocalSandbox(repo, runtime.config.sandbox, runtime.telemetry)
    summary = asyncio.run(run_autocover(runtime, repo, "ticket_price.py", sandbox=box,
                                        write=False, functions=["ticket_price"]))
    assert summary["mutation"] is None and llm.calls["fixer"] == []
    assert summary["frozen"] == 1          # the wrong boundary test, no repairs allowed
    assert summary["final"]["line_pct"] < 100  # group_total not targeted
