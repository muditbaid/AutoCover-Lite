"""End-to-end run of the agent graph with a scripted LLM and the local sandbox."""

import asyncio
import json
import re
import shutil
from pathlib import Path

import pytest

from autocover.config import Config, LLMConfig, RunConfig, SandboxConfig, TelemetryConfig
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


def test_ticket_price__wrong_boundary():
    assert ticket_price(12) == 5  # wrong: 12 is not a child


def test_ticket_price__another_child():
    assert ticket_price(3) == 5  # passes but adds no new coverage
```'''

GROUP_ROUND1 = '''```python
from ticket_price import group_total


def test_group_total__small_group():
    assert group_total([30, 30]) == 20.0
```'''

GROUP_ROUND2 = '''```python
from ticket_price import group_total


def test_group_total__discount():
    assert group_total([30] * 5, discount=0.1) == 45.0
```'''


class ScriptedLLM:
    def __init__(self):
        self.writer_prompts: list[str] = []
        self.group_calls = 0

    async def __call__(self, *, model, messages, **kwargs):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "SCENARIO_PLANNER" in system:
            fn = re.search(r"scenarios for `([\w.]+)`", user).group(1)
            text = json.dumps({"scenarios": SCENARIOS[fn]})
        else:
            self.writer_prompts.append(user)
            function_part = user.split("Function under test")[-1]
            if "test_group_total__" in user or "group_total" in function_part:
                self.group_calls += 1
                text = GROUP_ROUND1 if self.group_calls == 1 else GROUP_ROUND2
            else:
                text = TICKET_TESTS
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}}


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "repo"
    shutil.copytree(EXAMPLE, dest)
    return dest


def make_runtime(tmp_path, llm):
    config = Config(
        llm=LLMConfig(roles={"preparer": ["fake/planner"], "generator": ["fake/writer"]}),
        sandbox=SandboxConfig(backend="local", workdir=str(tmp_path / "ws"), max_parallel=4),
        run=RunConfig(max_rounds=3),
        telemetry=TelemetryConfig(jsonl_path=None),
    )
    telemetry = Telemetry()
    router = LLMRouter(config.llm, completion_fn=llm, require_keys=False, telemetry=telemetry)
    return Runtime(config=config, telemetry=telemetry, router=router, cache=None,
                   usage=UsageLedger())


def test_full_run_reaches_full_coverage(tmp_path, repo):
    llm = ScriptedLLM()
    runtime = make_runtime(tmp_path, llm)
    box = LocalSandbox(repo, runtime.config.sandbox, runtime.telemetry)
    summary = asyncio.run(run_autocover(runtime, repo, "ticket_price.py", sandbox=box))

    assert summary["final"]["line_pct"] == 100.0
    assert summary["final"]["branch_pct"] == 100.0
    assert summary["baseline"]["line_pct"] < 50  # only module-level lines before
    assert summary["rounds"] == 2                 # round 2 only for group_total's gap
    assert summary["failed"] == 1                 # the wrong boundary test
    assert summary["rejected_no_gain"] >= 1       # the redundant child test
    assert summary["removed_in_suite_check"] == []
    assert summary["llm"]["fake/writer"]["calls"] == 3

    # Round 2 was told exactly which lines are still uncovered.
    round2 = llm.writer_prompts[-1]
    assert "still not executed" in round2 and "total = total * (1 - discount)" in round2

    # The written suite passes on its own and keeps only signal-adding tests.
    written = (repo / "tests" / "test_ticket_price_autocover.py").read_text()
    assert "test_ticket_price__wrong_boundary" not in written
    assert "test_ticket_price__another_child" not in written
    assert "test_group_total__discount" in written
    assert written.count("import pytest") == 1
    check = box.run(RunRequest(target="ticket_price.py", tests={"test_final.py": written}))
    assert check.passed and len(check.tests) == summary["tests_in_suite"]


def test_dry_run_writes_nothing(tmp_path, repo):
    runtime = make_runtime(tmp_path, ScriptedLLM())
    box = LocalSandbox(repo, runtime.config.sandbox, runtime.telemetry)
    summary = asyncio.run(run_autocover(runtime, repo, "ticket_price.py", sandbox=box,
                                        write=False, functions=["ticket_price"]))
    assert summary["tests_in_suite"] == 4
    assert not (repo / "tests").exists()
