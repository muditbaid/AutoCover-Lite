"""Executed *inside* the sandbox (container or subprocess) - stdlib + pytest + coverage only.

Runs pytest on a directory of candidate tests while measuring line and branch coverage of
exactly one module, then writes a single JSON result:

    {"exit_code", "duration_s", "tests": [...], "collection_errors": [...],
     "coverage": {"executed_lines", "missing_lines", "executed_branches", "missing_branches"}}
"""

import argparse
import json
import os
import sys
import time


def _message(report):
    crash = getattr(getattr(report, "longrepr", None), "reprcrash", None)
    short = getattr(crash, "message", "") if crash else ""
    details = str(report.longrepr) if report.longrepr else ""
    return short[:500], details[-3000:]


class Collector:
    def __init__(self):
        self.tests = {}
        self.collection_errors = []

    def pytest_runtest_logreport(self, report):
        if report.when != "call" and report.outcome == "passed":
            return
        outcome = "error" if report.when != "call" and report.failed else report.outcome
        previous = self.tests.get(report.nodeid)
        if previous and previous["outcome"] in ("failed", "error"):
            return  # keep the first failure (e.g. setup error over teardown noise)
        short, details = _message(report)
        self.tests[report.nodeid] = {
            "nodeid": report.nodeid, "outcome": outcome, "message": short,
            "details": details, "duration": round(report.duration, 4),
        }

    def pytest_collectreport(self, report):
        if report.failed:
            short, details = _message(report)
            self.collection_errors.append(
                {"nodeid": report.nodeid or "<collection>", "message": short or details[-500:],
                 "details": details}
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--include", required=True, help="path of the module under test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    for path in (args.repo, os.path.join(args.repo, "src")):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    import coverage
    import pytest

    include = os.path.abspath(args.include)
    cov = coverage.Coverage(data_file=None, branch=True, include=[include], config_file=False)
    collector = Collector()
    start = time.perf_counter()
    cov.start()
    try:
        exit_code = int(pytest.main(
            [args.tests, "-q", "-p", "no:cacheprovider", "--rootdir", args.tests, "--no-header"],
            plugins=[collector],
        ))
    finally:
        cov.stop()
    duration = time.perf_counter() - start

    result = {
        "exit_code": exit_code,
        "duration_s": round(duration, 3),
        "tests": list(collector.tests.values()),
        "collection_errors": collector.collection_errors,
        "coverage": _coverage(cov, include, os.path.dirname(os.path.abspath(args.out))),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    return 0


def _coverage(cov, include, scratch):
    report_path = os.path.join(scratch, "coverage.json")
    try:
        cov.json_report(outfile=report_path, include=[include])
        with open(report_path, encoding="utf-8") as fh:
            files = json.load(fh)["files"]
        entry = next(iter(files.values()), None)
    except Exception:  # NoDataError when the module was never imported
        entry = None
    if entry is None:
        try:
            _, statements, _, missing, _ = cov.analysis2(include)
        except Exception:
            statements, missing = [], []
        return {"executed_lines": [], "missing_lines": sorted(missing or statements),
                "executed_branches": [], "missing_branches": [], "measured": False}
    return {
        "executed_lines": entry.get("executed_lines", []),
        "missing_lines": entry.get("missing_lines", []),
        "executed_branches": entry.get("executed_branches", []),
        "missing_branches": entry.get("missing_branches", []),
        "measured": True,
    }


if __name__ == "__main__":
    sys.exit(main())
