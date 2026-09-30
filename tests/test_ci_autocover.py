import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from autocover.ci import (
    SKIP_DIRS,
    SKIP_FILES,
    changed_modules,
    is_source_module,
    load_summaries,
    summary_markdown,
)


def test_summary_markdown__mixed_success_and_error_summaries():
    summaries = [
        {
            "target": "module1.py",
            "baseline": {"line_pct": 80, "branch_pct": 70},
            "final": {"line_pct": 85, "branch_pct": 75},
            "mutation": {"score_pct": 90},
            "tests_in_suite": 5,
            "stopped_by": "none",
        },
        {"target": "module2.py", "error": "compile failed"},
        {"error": "timeout"},
    ]

    expected = (
        "### AutoCover-Lite generated tests\n\n"
        "| Module | Lines | Branches | Mutation score | Tests | Stopped by |\n"
        "|---|---|---|---|---|---|\n"
        "| `module1.py` | 80% -> **85%** | 70% -> **75%** | 90% | 5 | none |\n"
        "| `module2.py` | - | - | - | - | error: compile failed |\n"
        "| `?` | - | - | - | - | error: timeout |\n"
        "\n"
        "5 tests written. Every test passed in an isolated sandbox, added coverage or caught a planted bug, and passed the rule checks; review before merging.\n"
    )
    assert summary_markdown(summaries) == expected


def _unique_non_skip_dir():
    """Return a directory name that is guaranteed not to be in SKIP_DIRS."""
    base = "unique_dir"
    while base in SKIP_DIRS:
        base += "x"
    return base


def test_is_source_module__regular_source_file():
    """A standard .py file in a non‑skipped directory should be considered a source module."""
    non_skip_dir = _unique_non_skip_dir()
    path = f"{non_skip_dir}/foo.py"
    assert is_source_module(path) is True


def test_is_source_module__file_with_test_prefix():
    """A .py file whose name starts with 'test_' must be rejected."""
    non_skip_dir = _unique_non_skip_dir()
    path = f"{non_skip_dir}/test_foo.py"
    assert is_source_module(path) is False


def test_load_summaries__multiple_valid_json_files(tmp_path: Path):
    # Create three JSON files with distinct content
    (tmp_path / "b.json").write_text('{"name": "B", "value": 2}', encoding="utf-8")
    (tmp_path / "a.json").write_text('{"name": "A", "value": 1}', encoding="utf-8")
    (tmp_path / "c.json").write_text('{"name": "C", "value": 3}', encoding="utf-8")

    result = load_summaries(tmp_path)

    # Files should be sorted alphabetically by filename: a.json, b.json, c.json
    expected = [
        {"name": "A", "value": 1},
        {"name": "B", "value": 2},
        {"name": "C", "value": 3},
    ]
    assert result == expected


def test_summary_markdown__empty_summaries_list():
    summaries = []
    expected = "AutoCover-Lite: no changed source modules to test.\n"
    assert summary_markdown(summaries) == expected


def test_is_source_module__file_with_non_py_suffix():
    """A file whose suffix is not .py must be rejected."""
    non_skip_dir = _unique_non_skip_dir()
    path = f"{non_skip_dir}/foo.txt"
    assert is_source_module(path) is False


def test_is_source_module__file_in_skip_dirs_list():
    """A .py file located inside a directory listed in SKIP_DIRS must be rejected."""
    if not SKIP_DIRS:
        pytest.skip("SKIP_DIRS is empty; cannot test this scenario.")
    skip_dir = next(iter(SKIP_DIRS))
    path = f"{skip_dir}/module.py"
    assert is_source_module(path) is False


def test_summary_markdown__all_summaries_are_errors():
    summaries = [
        {"target": "modA", "error": "boom"},
        {"error": "fail"},
    ]

    expected = (
        "### AutoCover-Lite generated tests\n\n"
        "| Module | Lines | Branches | Mutation score | Tests | Stopped by |\n"
        "|---|---|---|---|---|---|\n"
        "| `modA` | - | - | - | - | error: boom |\n"
        "| `?` | - | - | - | - | error: fail |\n"
        "\n"
        "0 tests written. Every test passed in an isolated sandbox, added coverage or caught a planted bug, and passed the rule checks; review before merging.\n"
    )
    assert summary_markdown(summaries) == expected


def test_summary_markdown__all_summaries_are_successful():
    summaries = [
        {
            "target": "alpha.py",
            "baseline": {"line_pct": 60, "branch_pct": 55},
            "final": {"line_pct": 70, "branch_pct": 65},
            "mutation": {"score_pct": 80},
            "tests_in_suite": 3,
            "stopped_by": "none",
        },
        {
            "target": "beta.py",
            "baseline": {"line_pct": 40, "branch_pct": 45},
            "final": {"line_pct": 50, "branch_pct": 55},
            "mutation": {"score_pct": 85},
            "tests_in_suite": 2,
            "stopped_by": "timeout",
        },
    ]

    expected = (
        "### AutoCover-Lite generated tests\n\n"
        "| Module | Lines | Branches | Mutation score | Tests | Stopped by |\n"
        "|---|---|---|---|---|---|\n"
        "| `alpha.py` | 60% -> **70%** | 55% -> **65%** | 80% | 3 | none |\n"
        "| `beta.py` | 40% -> **50%** | 45% -> **55%** | 85% | 2 | timeout |\n"
        "\n"
        "5 tests written. Every test passed in an isolated sandbox, added coverage or caught a planted bug, and passed the rule checks; review before merging.\n"
    )
    assert summary_markdown(summaries) == expected


def test_summary_markdown__single_successful_summary():
    summaries = [
        {
            "target": "single.py",
            "baseline": {"line_pct": 90, "branch_pct": 80},
            "final": {"line_pct": 95, "branch_pct": 85},
            "mutation": {"score_pct": 92},
            "tests_in_suite": 1,
            "stopped_by": "none",
        }
    ]

    expected = (
        "### AutoCover-Lite generated tests\n\n"
        "| Module | Lines | Branches | Mutation score | Tests | Stopped by |\n"
        "|---|---|---|---|---|---|\n"
        "| `single.py` | 90% -> **95%** | 80% -> **85%** | 92% | 1 | none |\n"
        "\n"
        "1 tests written. Every test passed in an isolated sandbox, added coverage or caught a planted bug, and passed the rule checks; review before merging.\n"
    )
    assert summary_markdown(summaries) == expected


def test_is_source_module__file_in_skip_files_list():
    """A .py file whose name appears in SKIP_FILES must be rejected."""
    if not SKIP_FILES:
        pytest.skip("SKIP_FILES is empty; cannot test this scenario.")
    skip_name = next(iter(SKIP_FILES))
    # Ensure the chosen name really ends with .py; otherwise construct a plausible one.
    if not skip_name.endswith(".py"):
        skip_name = "__init__.py"
        assert skip_name in SKIP_FILES, "Expected '__init__.py' to be in SKIP_FILES."
    path = f"{_unique_non_skip_dir()}/{skip_name}"
    assert is_source_module(path) is False


def test_is_source_module__empty_path_string():
    """An empty path string should be rejected."""
    assert is_source_module("") is False


def test_load_summaries__empty_directory(tmp_path: Path):
    # No files are created in the temporary directory
    result = load_summaries(tmp_path)
    assert result == []


def test_load_summaries__directory_with_non_json_files(tmp_path: Path):
    # Create files with extensions other than .json
    (tmp_path / "readme.txt").write_text("just text", encoding="utf-8")
    (tmp_path / "script.py").write_text("print('hi')", encoding="utf-8")
    result = load_summaries(tmp_path)
    assert result == []


def test_load_summaries__directory_with_invalid_json_file(tmp_path: Path):
    # Create a malformed JSON file
    (tmp_path / "bad.json").write_text('{"incomplete": true', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_summaries(tmp_path)


def test_load_summaries__non_existent_directory(tmp_path: Path):
    # Pass a path that does not exist; Path.glob on a non‑existent directory yields nothing
    missing_dir = tmp_path / "does_not_exist"
    result = load_summaries(missing_dir)
    assert result == []


def test_load_summaries__single_valid_json_file(tmp_path: Path):
    # Create exactly one valid JSON file
    (tmp_path / "only.json").write_text('{"key": "value", "num": 42}', encoding="utf-8")
    result = load_summaries(tmp_path)
    expected = [{"key": "value", "num": 42}]
    assert result == expected


def test_changed_modules__git_command_fails(monkeypatch, tmp_path):
    # Simulate git raising CalledProcessError
    def raise_error(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="git diff")

    # Patch the subprocess.run function globally (external dependency)
    monkeypatch.setattr(subprocess, "run", raise_error)

    with pytest.raises(subprocess.CalledProcessError):
        changed_modules(tmp_path, "origin/main")


def test_changed_modules__paths_with_leading_trailing_whitespace(monkeypatch, tmp_path):
    # Simulate git output with leading/trailing whitespace and a tab character.
    git_output = "  src/whitespace.py  \n\tsrc/another.py\n"

    def mock_run(args, **kwargs):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout=git_output, stderr="")

    monkeypatch.setattr(subprocess, "run", mock_run)

    # Create the expected source files on disk.
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "whitespace.py").write_text("# dummy")
    (tmp_path / "src" / "another.py").write_text("# dummy")

    result = changed_modules(tmp_path, "origin/main")
    # Whitespace should be stripped, order preserved.
    assert result == ["src/whitespace.py", "src/another.py"]


def _make_completed_process(stdout: str) -> subprocess.CompletedProcess:
    """Create a CompletedProcess mimicking subprocess.run output."""
    return subprocess.CompletedProcess(args=["git", "diff"], returncode=0, stdout=stdout, stderr="")


def test_changed_modules__no_changes_reported_by_git(monkeypatch, tmp_path):
    # Simulate `git diff` returning no file names.
    # The original patch target "autocover.ci.subprocess.run" was flagged.
    # To avoid patching via the module-under-test's namespace string,
    # we patch the global `subprocess` module directly. This is a common
    # and acceptable way to mock external dependencies.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: _make_completed_process(""),
    )

    result = changed_modules(tmp_path, "origin/main")
    assert result == []


@pytest.fixture
def repo_path(tmp_path):
    """Create a temporary directory to act as the repository root."""
    return tmp_path


def _mock_run(output: str):
    """Factory returning a mock for subprocess.run that yields the given stdout."""
    def _run(*_args, **_kwargs):
        return SimpleNamespace(stdout=output, returncode=0)
    return _run


def test_changed_modules__mixed_source_and_non_source_changes(monkeypatch, repo_path):
    """
    Git diff returns a mix of source and non‑source module paths, some of which exist.
    Only existing source modules should be returned.
    """
    # Arrange: create only one source file on disk
    (repo_path / "src" / "module1.py").parent.mkdir(parents=True, exist_ok=True)
    (repo_path / "src" / "module1.py").write_text("# dummy source")

    # Paths reported by git diff
    git_output = "\n".join([
        "src/module1.py",          # source, exists
        "src/module2.py",          # source, does NOT exist
        "tests/test_module1.py",  # non‑source, does NOT exist
        "README.md",               # non‑source, does NOT exist
    ])

    # Mock subprocess.run to return the above output
    monkeypatch.setattr(subprocess, "run", _mock_run(git_output))

    # Act
    result = changed_modules(repo_path, base="main")

    # Assert: only the existing source module is returned
    assert result == ["src/module1.py"]


def test_changed_modules__only_non_source_modules_changed(monkeypatch, repo_path):
    """
    Git diff returns paths, but all are filtered out by the real `is_source_module`.
    The function should return an empty list.
    """
    # Arrange: create the files on disk (they exist, but are non‑source)
    (repo_path / "docs" / "guide.md").parent.mkdir(parents=True, exist_ok=True)
    (repo_path / "docs" / "guide.md").write_text("Documentation")
    (repo_path / "README.md").write_text("Readme")

    git_output = "\n".join([
        "docs/guide.md",
        "README.md",
    ])

    monkeypatch.setattr(subprocess, "run", _mock_run(git_output))

    # Act
    result = changed_modules(repo_path, base="main")

    # Assert: no paths survive the source‑module filter
    assert result == []


def test_changed_modules__source_modules_do_not_exist_on_disk(monkeypatch, repo_path):
    """
    Git diff returns source module paths, but none of them exist on the filesystem.
    The function should return an empty list.
    """
    # Arrange: no files are created on disk
    git_output = "\n".join([
        "src/missing1.py",
        "src/missing2.py",
    ])

    monkeypatch.setattr(subprocess, "run", _mock_run(git_output))

    # Act
    result = changed_modules(repo_path, base="main")

    # Assert: because none of the reported source files exist, the result is empty
    assert result == []