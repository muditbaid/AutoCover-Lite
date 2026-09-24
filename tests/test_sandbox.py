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


def test_non_ascii_test_source_and_output(sandbox):
    src = ('from ticket_price import ticket_price\n\n'
           'def test_unicode():\n    """Age‑boundary – café."""\n'
           '    print("résumé ‑")\n    assert ticket_price(5) == 6, "über"\n')
    result = sandbox.run(RunRequest(target=TARGET, tests={"test_uni.py": src}))
    assert result.status == "ok" and not result.passed
    assert "über" in result.diagnostics()


GROUP = """
import pytest
from ticket_price import group_total


@pytest.fixture
def five_adults():
    return [30] * 5


def test_group_discount(five_adults):
    assert group_total(five_adults, discount=0.1) == 45.0


@pytest.mark.parametrize("ages,total", [([30], 10.0), ([5, 70], 12.0)])
def test_group_small(ages, total):
    assert group_total(ages) == total
"""


def test_batched_run_attributes_coverage_exactly_per_file(sandbox):
    files = {"test_good.py": GOOD, "test_bad.py": BAD, "test_group.py": GROUP,
             "test_broken.py": BROKEN}
    batch = sandbox.run(RunRequest(target=TARGET, tests=files, per_test=True))
    assert batch.status == "ok" and batch.per_test
    for name, source in files.items():
        alone = sandbox.run(RunRequest(target=TARGET, tests={name: source}))
        part = batch.for_file(name)
        assert part.passed == alone.passed, name
        assert [t.outcome for t in part.tests] == [t.outcome for t in alone.tests], name
        assert bool(part.collection_errors) == bool(alone.collection_errors), name
        if alone.tests:
            assert part.coverage.executed_lines == alone.coverage.executed_lines - \
                _import_lines(sandbox), name
            assert part.coverage.executed_branches == alone.coverage.executed_branches, name
        assert part.coverage.all_lines == batch.coverage.all_lines


def _import_lines(sandbox):
    """Module-level lines run at import time belong to no test in a batched run."""
    probe = sandbox.run(RunRequest(target=TARGET, tests={
        "test_probe.py": "import ticket_price\n\ndef test_p():\n    assert ticket_price\n"}))
    return probe.coverage.executed_lines


def test_transient_engine_errors_are_retried_then_reported(tmp_path, monkeypatch):
    import autocover.tools.sandbox as sb

    monkeypatch.setattr(sb.time, "sleep", lambda s: None)
    box = DockerSandbox(EXAMPLE, SandboxConfig(workdir=str(tmp_path)), client=object())
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError(231, "CreateFile", "All pipe instances are busy.")
        return "ok"

    assert box._api(flaky) == "ok" and len(calls) == 3

    def always_busy():
        raise OSError(231, "CreateFile", "All pipe instances are busy.")

    with pytest.raises(sb.SandboxUnavailable):
        box._api(always_busy)
    with pytest.raises(ValueError):  # real errors are not retried
        box._api(lambda: (_ for _ in ()).throw(ValueError("bad request")))
