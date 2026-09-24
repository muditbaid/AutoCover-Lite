"""Helpers for running AutoCover-Lite in CI (used by the GitHub Action in action.yml)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path, PurePosixPath

SKIP_DIRS = {"tests", "test", "testing", "docs", "doc", "examples", "migrations", "scripts",
             "benchmarks", "bench", ".github"}
SKIP_FILES = {"conftest.py", "setup.py", "noxfile.py", "manage.py", "__main__.py"}


def is_source_module(path: str) -> bool:
    """A .py file worth generating tests for (not a test, config or tooling file)."""
    p = PurePosixPath(path)
    if p.suffix != ".py" or p.name in SKIP_FILES:
        return False
    if p.name.startswith("test_") or p.name.endswith("_test.py"):
        return False
    return not (set(p.parts[:-1]) & SKIP_DIRS)


def changed_modules(repo: str | Path, base: str) -> list[str]:
    """Source modules added or modified on this branch relative to `base`."""
    out = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", f"{base}...HEAD"],
        cwd=repo, capture_output=True, text=True, check=True).stdout
    paths = [line.strip() for line in out.splitlines() if line.strip()]
    return [p for p in paths if is_source_module(p) and (Path(repo) / p).exists()]


def summary_markdown(summaries: list[dict]) -> str:
    """Markdown table for the job summary / PR body."""
    if not summaries:
        return "AutoCover-Lite: no changed source modules to test.\n"
    rows = ["| Module | Lines | Branches | Mutation score | Tests | Stopped by |",
            "|---|---|---|---|---|---|"]
    for s in summaries:
        if "error" in s:
            rows.append(f"| `{s.get('target', '?')}` | - | - | - | - | error: {s['error']} |")
            continue
        base, final = s["baseline"], s["final"]
        mut = s.get("mutation") or {}
        rows.append(
            f"| `{s['target']}` | {base['line_pct']}% -> **{final['line_pct']}%** | "
            f"{base['branch_pct']}% -> **{final['branch_pct']}%** | "
            f"{mut.get('score_pct', '-')}% | {s['tests_in_suite']} | {s['stopped_by']} |")
    total = sum(s.get("tests_in_suite", 0) for s in summaries)
    return ("### AutoCover-Lite generated tests\n\n" + "\n".join(rows) +
            f"\n\n{total} tests written. Every test passed in an isolated sandbox, added "
            "coverage or caught a planted bug, and passed the rule checks; review before "
            "merging.\n")


def load_summaries(directory: str | Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(Path(directory).glob("*.json"))]
