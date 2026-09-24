"""Data passed between agents.

`RunState` is the LangGraph state (plain data that changes every step); `RunContext`
holds the long-lived services and bookkeeping objects the agent nodes share.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, Field

from autocover.config import Config
from autocover.llm.router import LLMRouter
from autocover.telemetry import Telemetry
from autocover.tools.context import ModuleContext
from autocover.tools.coverage_runner import CoverageTracker
from autocover.tools.sandbox import Sandbox


class Scenario(BaseModel):
    id: str            # identifier-safe, used in test names: test_<func>__<id>
    function: str      # qualname of the function under test
    description: str
    kind: str = "happy"  # happy | edge | error


CandidateStatus = Literal["pending", "passed", "failed", "needs_fix", "accepted", "rejected",
                          "frozen"]


class Candidate(BaseModel):
    """One generated test case, runnable on its own (shared preamble + one test)."""

    id: str
    function: str
    test_name: str
    scenario_id: str | None = None
    code: str
    round: int
    model: str = ""
    status: CandidateStatus = "pending"
    reason: str = ""
    diagnostics: str = ""
    new_lines: list[int] = Field(default_factory=list)
    new_branches: list[tuple[int, int]] = Field(default_factory=list)
    duration_s: float = 0.0
    attempt: int = 0                  # 0 = as generated, n = after n Fixer repairs
    parent_id: str | None = None      # candidate this one was repaired from
    violations: list[str] = Field(default_factory=list)  # rule ids (errors and warnings)
    mutants_run: int = 0
    killed: list[str] = Field(default_factory=list)      # mutant ids this test kills
    new_kills: list[str] = Field(default_factory=list)   # kills no accepted test had
    accepted_by: str = ""             # coverage | mutants | scenario


class RunState(TypedDict, total=False):
    round: int
    scenarios: dict[str, list[Scenario]]   # function qualname -> scenarios
    targets: list[str]                     # functions to (re)generate for, by priority
    pending: list[Candidate]               # produced by Generator/Fixer, run by Executor
    executed: list[Candidate]              # run by Executor, judged by Validator
    to_fix: list[Candidate]                # sent by Validator to the Fixer
    history: list[Candidate]               # every executed candidate
    suite: str                             # accumulated test file source
    feedback: dict[str, str]               # function -> failures to show the next round
    final: dict                            # summary written by the finalize node


@dataclass
class RunContext:
    config: Config
    repo: Path
    target: str        # module under test, repo-relative
    test_path: str     # output test file, repo-relative
    router: LLMRouter
    sandbox: Sandbox
    telemetry: Telemetry
    module: ModuleContext
    tracker: CoverageTracker = field(default_factory=CoverageTracker)
    deadline: float = 0.0
    functions: list[str] | None = None   # restrict to these qualnames
    seen_code: set[str] = field(default_factory=set)  # dedup of tried candidates
    covered_scenarios: set[tuple[str, str]] = field(default_factory=set)
    baseline: dict = field(default_factory=dict)  # coverage summary before generation
    results: dict = field(default_factory=dict)   # candidate id -> RunResult
    mutant_pool: list | None = None                # Mutant objects, built lazily
    killed_mutants: set[str] = field(default_factory=set)  # killed by accepted tests
    survivors: dict = field(default_factory=dict)  # candidate id -> mutants it let survive
    scenarios: dict = field(default_factory=dict)  # function -> scenarios (from Preparer)
    _counter: int = 0

    def next_id(self) -> str:
        self._counter += 1
        return f"c{self._counter}"

    def time_left(self) -> float:
        return self.deadline - time.monotonic()
