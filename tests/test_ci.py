import json
import subprocess

import yaml
from typer.testing import CliRunner

from autocover.ci import changed_modules, is_source_module, summary_markdown
from autocover.cli import app
from tests.test_config import ROOT


def test_is_source_module():
    assert is_source_module("src/pkg/core.py")
    assert is_source_module("pkg/__init__.py")
    for path in ("tests/test_core.py", "pkg/tests/helpers.py", "pkg/test_x.py", "pkg/x_test.py",
                 "conftest.py", "setup.py", "docs/conf.py", "pkg/data.json", "bench/run.py"):
        assert not is_source_module(path), path


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_changed_modules_lists_only_added_or_modified_sources(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "pkg" / "a.py").write_text("def a():\n    return 1\n")
    (repo / "pkg" / "gone.py").write_text("x = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "pkg" / "a.py").write_text("def a():\n    return 2\n")
    (repo / "pkg" / "b.py").write_text("def b():\n    return 3\n")
    (repo / "tests").mkdir()
    (repo / "tests" / "test_a.py").write_text("def test_a():\n    pass\n")
    (repo / "pkg" / "gone.py").unlink()
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature")
    assert changed_modules(repo, "main") == ["pkg/a.py", "pkg/b.py"]


def test_summary_markdown_and_cli(tmp_path):
    ok = {"target": "pkg/a.py", "baseline": {"line_pct": 10.0, "branch_pct": 0.0},
          "final": {"line_pct": 95.0, "branch_pct": 90.0}, "mutation": {"score_pct": 80.0},
          "tests_in_suite": 7, "stopped_by": "no gaps left"}
    bad = {"target": "pkg/b.py", "error": "run failed, see log"}
    (tmp_path / "a.json").write_text(json.dumps(ok))
    (tmp_path / "b.json").write_text(json.dumps(bad))
    text = CliRunner().invoke(app, ["ci-summary", str(tmp_path)]).output
    assert "10.0% -> **95.0%**" in text and "80.0%" in text and "7 tests written" in text
    assert "error: run failed" in text
    assert "no changed source modules" in summary_markdown([])


def test_action_definition_is_valid():
    action = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))
    assert action["runs"]["using"] == "composite"
    assert {"base", "paths", "budget-min", "max-llm-calls", "create-pr"} <= set(action["inputs"])
    steps = [s.get("name") or s.get("uses") for s in action["runs"]["steps"]]
    assert "Generate tests" in steps and "peter-evans/create-pull-request@v6" in \
        [s.get("uses") for s in action["runs"]["steps"]]
    workflow = yaml.safe_load((ROOT / "examples" / "workflows" / "autocover.yml").read_text())
    assert workflow["permissions"] == {"contents": "write", "pull-requests": "write"}
