"""Isolated test execution.

Every run gets a fresh workspace holding the candidate test files and the sandbox runner.
Runs that override repo files (mutants) also get a private copy of the repo, so parallel
runs never see each other's changes.

* DockerSandbox - the real backend. One image per target repo (deps + pytest + coverage,
  cached by a hash of the dependency files); each run is its own container with
  `--network none`, memory / CPU / pid limits and a hard timeout.
* LocalSandbox - a subprocess with the current interpreter. No isolation; for development
  and unit tests on machines without Docker.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal

from autocover.config import SandboxConfig
from autocover.telemetry import Telemetry

RUNNER = Path(__file__).with_name("sandbox_runner.py")
IGNORED = shutil.ignore_patterns(
    ".git", ".venv", "venv", "__pycache__", "*.pyc", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", "node_modules", ".autocover", ".tox", "build", "dist", "*.egg-info",
)
DEPENDENCY_FILES = ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
                    "requirements-dev.txt", "requirements-test.txt")

DOCKERFILE = """\
FROM python:{python_version}-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
RUN pip install "pytest>=8" "coverage>=7.5"
COPY . /repo
# Install the project's dependencies. The project itself is installed editable so that
# at run time imports resolve to the (mounted) /repo sources that coverage measures.
RUN set -e; cd /repo; \\
    for f in requirements.txt requirements-dev.txt requirements-test.txt; do \\
      if [ -f "$f" ]; then pip install -r "$f"; fi; done; \\
    if [ -f pyproject.toml ] || [ -f setup.py ]; then \\
      pip install -e . || echo "editable install failed; falling back to PYTHONPATH"; fi
WORKDIR /ws
"""


@dataclass
class TestOutcome:
    nodeid: str
    outcome: str  # passed | failed | error | skipped
    message: str = ""
    details: str = ""
    duration: float = 0.0


@dataclass(frozen=True)
class CoverageData:
    executed_lines: frozenset[int] = frozenset()
    missing_lines: frozenset[int] = frozenset()
    executed_branches: frozenset[tuple[int, int]] = frozenset()
    missing_branches: frozenset[tuple[int, int]] = frozenset()
    measured: bool = True

    @property
    def all_lines(self) -> frozenset[int]:
        return self.executed_lines | self.missing_lines

    @property
    def all_branches(self) -> frozenset[tuple[int, int]]:
        return self.executed_branches | self.missing_branches

    @classmethod
    def from_json(cls, data: dict) -> CoverageData:
        return cls(
            executed_lines=frozenset(data.get("executed_lines", [])),
            missing_lines=frozenset(data.get("missing_lines", [])),
            executed_branches=frozenset(tuple(b) for b in data.get("executed_branches", [])),
            missing_branches=frozenset(tuple(b) for b in data.get("missing_branches", [])),
            measured=data.get("measured", True),
        )


@dataclass
class RunResult:
    status: Literal["ok", "timeout", "crashed"]
    exit_code: int | None = None
    tests: list[TestOutcome] = field(default_factory=list)
    collection_errors: list[dict] = field(default_factory=list)
    coverage: CoverageData | None = None
    log_tail: str = ""
    duration_s: float = 0.0

    @property
    def passed(self) -> bool:
        return (self.status == "ok" and not self.collection_errors and bool(self.tests)
                and all(t.outcome in ("passed", "skipped") for t in self.tests)
                and any(t.outcome == "passed" for t in self.tests))

    @property
    def failures(self) -> list[TestOutcome]:
        return [t for t in self.tests if t.outcome in ("failed", "error")]

    def diagnostics(self, limit: int = 4000) -> str:
        """Human/LLM-readable summary of what went wrong."""
        if self.status == "timeout":
            return "Test run timed out (possible infinite loop, sleep or blocking I/O)."
        parts = [f"Collection error in {e['nodeid']}:\n{e['details'] or e['message']}"
                 for e in self.collection_errors]
        parts += [f"{t.outcome.upper()} {t.nodeid}: {t.message}\n{t.details}"
                  for t in self.failures]
        if not parts and self.status == "crashed":
            parts.append(f"Sandbox crashed:\n{self.log_tail}")
        if not parts and not self.tests:
            parts.append("No tests were collected.")
        return "\n\n".join(parts)[-limit:]


@dataclass
class RunRequest:
    target: str                      # module under test, repo-relative ("src/pkg/mod.py")
    tests: dict[str, str]            # test filename -> source
    overrides: dict[str, str] = field(default_factory=dict)  # repo-relative path -> source
    timeout_s: float | None = None
    label: str = ""


class Sandbox(ABC):
    backend = "abstract"

    def __init__(self, repo_root: str | Path, config: SandboxConfig,
                 telemetry: Telemetry | None = None):
        self.repo_root = Path(repo_root).resolve()
        self.config = config
        self.telemetry = telemetry or Telemetry()
        self._workroot = Path(config.workdir) if config.workdir else Path(tempfile.gettempdir())
        self._workroot.mkdir(parents=True, exist_ok=True)
        self._semaphore: asyncio.Semaphore | None = None

    def prepare(self) -> None:  # noqa: B027 - optional hook
        """One-time setup (e.g. building the image)."""

    @abstractmethod
    def _execute(self, repo_dir: Path, meta_dir: Path, target: str,
                 timeout: float) -> tuple[str, int | None, str]:
        """Run the runner; return (status, exit_code, log_tail)."""

    def run(self, request: RunRequest) -> RunResult:
        timeout = request.timeout_s or self.config.timeout_s
        ws = self._workroot / f"autocover-{uuid.uuid4().hex[:10]}"
        meta = ws / "meta"
        (meta / "tests").mkdir(parents=True)
        try:
            shutil.copy(RUNNER, meta / "runner.py")
            for name, source in request.tests.items():
                _write(meta / "tests" / name, source)
            repo_dir = self.repo_root
            if request.overrides:
                repo_dir = ws / "repo"
                shutil.copytree(self.repo_root, repo_dir, ignore=IGNORED)
                for rel, source in request.overrides.items():
                    _write(repo_dir / rel, source)
            with self.telemetry.span("sandbox", "run", backend=self.backend,
                                     label=request.label, mutated=bool(request.overrides)) as span:
                start = time.perf_counter()
                status, exit_code, log = self._execute(repo_dir, meta, request.target, timeout)
                result = self._read_result(meta / "result.json", status, exit_code, log)
                result.duration_s = round(time.perf_counter() - start, 3)
                span.update(result=result.status, passed=result.passed,
                            tests=len(result.tests))
            return result
        finally:
            shutil.rmtree(ws, ignore_errors=True)

    async def arun(self, request: RunRequest) -> RunResult:
        """Run in a worker thread, at most `max_parallel` at a time."""
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_parallel)
        async with self._semaphore:
            return await asyncio.to_thread(self.run, request)

    @staticmethod
    def _read_result(path: Path, status: str, exit_code: int | None, log: str) -> RunResult:
        if status != "ok" or not path.exists():
            return RunResult(status="timeout" if status == "timeout" else "crashed",
                             exit_code=exit_code, log_tail=log[-4000:])
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunResult(
            status="ok",
            exit_code=data["exit_code"],
            tests=[TestOutcome(**t) for t in data["tests"]],
            collection_errors=data["collection_errors"],
            coverage=CoverageData.from_json(data["coverage"]),
            log_tail=log[-4000:],
        )


class LocalSandbox(Sandbox):
    backend = "local"

    def _execute(self, repo_dir, meta_dir, target, timeout):
        env = {**os.environ, "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
        env["PYTHONPATH"] = os.pathsep.join([str(repo_dir), str(repo_dir / "src")])
        cmd = [sys.executable, str(meta_dir / "runner.py"), "--repo", str(repo_dir),
               "--tests", str(meta_dir / "tests"), "--include", str(repo_dir / target),
               "--out", str(meta_dir / "result.json")]
        try:
            proc = subprocess.run(cmd, cwd=meta_dir, env=env, capture_output=True, text=True,
                                  timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            return "timeout", None, _text(exc.stdout) + _text(exc.stderr)
        return "ok", proc.returncode, proc.stdout + proc.stderr


class DockerSandbox(Sandbox):
    backend = "docker"

    def __init__(self, repo_root, config, telemetry=None, client=None):
        super().__init__(repo_root, config, telemetry)
        self._client = client
        self.image: str | None = None

    @property
    def client(self):
        if self._client is None:
            import docker

            self._client = docker.from_env()
        return self._client

    def image_tag(self) -> str:
        digest = hashlib.sha256()
        digest.update(DOCKERFILE.format(python_version=self.config.python_version).encode())
        for name in DEPENDENCY_FILES:
            path = self.repo_root / name
            if path.exists():
                digest.update(name.encode() + path.read_bytes())
        return f"autocover-sandbox:{digest.hexdigest()[:12]}"

    def prepare(self) -> None:
        import docker.errors

        tag = self.image_tag()
        try:
            self.client.images.get(tag)
        except docker.errors.ImageNotFound:
            with self.telemetry.span("sandbox", "build_image", tag=tag), \
                    tempfile.TemporaryDirectory(dir=self._workroot) as ctx:
                context = Path(ctx) / "context"
                shutil.copytree(self.repo_root, context, ignore=IGNORED)
                (context / "Dockerfile").write_text(
                    DOCKERFILE.format(python_version=self.config.python_version),
                    encoding="utf-8", newline="\n",
                )
                self.client.images.build(path=str(context), tag=tag, rm=True, forcerm=True)
        self.image = tag

    def _execute(self, repo_dir, meta_dir, target, timeout):
        if self.image is None:
            self.prepare()
        include = str(PurePosixPath("/repo") / PurePosixPath(Path(target).as_posix()))
        container = self.client.containers.run(
            self.image,
            command=["python", "/ws/runner.py", "--repo", "/repo", "--tests", "/ws/tests",
                     "--include", include, "--out", "/ws/result.json"],
            volumes={
                str(repo_dir): {"bind": "/repo", "mode": "ro"},
                str(meta_dir): {"bind": "/ws", "mode": "rw"},
            },
            environment={"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
                         "PYTHONPATH": "/repo/src:/repo"},
            working_dir="/ws",
            network_mode="none",
            mem_limit=self.config.memory,
            nano_cpus=int(self.config.cpus * 1e9),
            pids_limit=self.config.pids_limit,
            detach=True,
        )
        try:
            try:
                exit_code = container.wait(timeout=timeout)["StatusCode"]
                status = "ok"
            except Exception:  # requests timeout while waiting
                container.kill()
                exit_code, status = None, "timeout"
            log = container.logs(tail=200).decode("utf-8", "replace")
            return status, exit_code, log
        finally:
            container.remove(force=True)


def make_sandbox(repo_root: str | Path, config: SandboxConfig,
                 telemetry: Telemetry | None = None) -> Sandbox:
    cls = DockerSandbox if config.backend == "docker" else LocalSandbox
    return cls(repo_root, config, telemetry)


def docker_available() -> bool:
    try:
        import docker

        docker.from_env(timeout=5).ping()
        return True
    except Exception:
        return False


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value
