from pathlib import Path

import pytest

from autocover.config import SandboxConfig
from autocover.tools.coverage_runner import CoverageTracker
from autocover.tools.mutator import generate_mutants
from autocover.tools.sandbox import (
    DockerSandbox,
    LocalSandbox,
    RunRequest,
    docker_available,
)

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ticket_price"
TARGET = "ticket_price.py"

GOOD = """
import pytest
from ticket_price import ticket_price

def test_child():
    assert ticket_price(5) == 5

def test_senior_boundary():
    assert ticket_price(65) == 7

def test_negative():
    with pytest.raises(ValueError):
        ticket_price(-1)
"""

BAD = """
from ticket_price import ticket_price

def test_wrong():
    assert ticket_price(30) == 999
"""

SLOW = """
import time

def test_hangs():
    time.sleep(60)
"""

NETWORK = """
import socket

def test_network():
    socket.create_connection(("1.1.1.1", 53), timeout=3).close()
"""

BROKEN = "def test_x(:\n    pass\n"

_HAS_DOCKER = docker_available()


def _backends():
    yield pytest.param("local", id="local")
    yield pytest.param("docker", id="docker", marks=[
        pytest.mark.docker,
        pytest.mark.skipif(not _HAS_DOCKER, reason="Docker engine not available"),
    ])


@pytest.fixture(scope="module", params=list(_backends()))
def sandbox(request, tmp_path_factory):
    config = SandboxConfig(backend=request.param, timeout_s=60, max_parallel=2,
                           workdir=str(tmp_path_factory.mktemp("ws")))
    cls = DockerSandbox if request.param == "docker" else LocalSandbox
    box = cls(EXAMPLE, config)
    box.prepare()
    return box


def test_passing_tests_report_coverage(sandbox):
    result = sandbox.run(RunRequest(target=TARGET, tests={"test_good.py": GOOD}))
    assert result.passed, result.diagnostics()
    assert {t.outcome for t in result.tests} == {"passed"} and len(result.tests) == 3
    cov = result.coverage
    assert cov.measured and cov.executed_lines and cov.missing_lines  # group_total untested
    assert cov.executed_branches


def test_failing_test_has_diagnostics(sandbox):
    result = sandbox.run(RunRequest(target=TARGET, tests={"test_bad.py": BAD}))
    assert not result.passed
    (failure,) = result.failures
    assert failure.outcome == "failed" and "999" in failure.message + failure.details
    assert "test_wrong" in result.diagnostics()


def test_collection_error_is_reported(sandbox):
    result = sandbox.run(RunRequest(target=TARGET, tests={"test_broken.py": BROKEN}))
    assert not result.passed and result.collection_errors
    assert "SyntaxError" in result.diagnostics() or "invalid syntax" in result.diagnostics()


def test_timeout(sandbox):
    result = sandbox.run(RunRequest(target=TARGET, tests={"test_slow.py": SLOW}, timeout_s=5))
    assert result.status == "timeout" and not result.passed


def test_mutant_override_is_killed_by_boundary_test(sandbox):
    source = (EXAMPLE / TARGET).read_text()
    mutant = next(m for m in generate_mutants(source, functions={"ticket_price"})
                  if m.description == ">= -> >")
    result = sandbox.run(RunRequest(target=TARGET, tests={"test_good.py": GOOD},
                                    overrides={TARGET: mutant.source}))
    assert result.status == "ok" and not result.passed  # boundary test catches it
    # The original repo was not modified by the override.
    assert (EXAMPLE / TARGET).read_text() == source


@pytest.mark.docker
@pytest.mark.skipif(not _HAS_DOCKER, reason="Docker engine not available")
def test_docker_sandbox_blocks_network(tmp_path):
    box = DockerSandbox(EXAMPLE, SandboxConfig(workdir=str(tmp_path), timeout_s=60))
    result = box.run(RunRequest(target=TARGET, tests={"test_net.py": NETWORK}))
    assert result.status == "ok" and not result.passed
    assert "network" in result.diagnostics().lower() or "OSError" in result.diagnostics()


def test_parallel_runs_are_isolated(sandbox):
    import asyncio

    async def both():
        return await asyncio.gather(
            sandbox.arun(RunRequest(target=TARGET, tests={"test_good.py": GOOD})),
            sandbox.arun(RunRequest(target=TARGET, tests={"test_bad.py": BAD})),
        )

    good, bad = asyncio.run(both())
    assert good.passed and not bad.passed


def test_coverage_tracker_gain(sandbox):
    first = sandbox.run(RunRequest(target=TARGET, tests={"test_good.py": GOOD}))
    tracker = CoverageTracker()
    gained = tracker.add(first.coverage)
    assert gained and tracker.line_pct > 0
    assert not tracker.gain(first.coverage)  # same run adds nothing new
    group = """
from ticket_price import group_total

def test_group_discount():
    assert group_total([30] * 5, discount=0.1) == 45.0
"""
    second = sandbox.run(RunRequest(target=TARGET, tests={"test_group.py": group}))
    assert second.passed, second.diagnostics()
    assert tracker.gain(second.coverage).lines
    tracker.add(second.coverage)
    assert tracker.summary()["lines_covered"] > len(first.coverage.executed_lines)
