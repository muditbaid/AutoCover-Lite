"""Executed *inside* the sandbox (container or subprocess) - stdlib + pytest + coverage only.

Runs pytest on a directory of candidate tests while measuring line and branch coverage of
exactly one module, then writes a single JSON result:

    {"exit_code", "duration_s", "tests": [...], "collection_errors": [...],
     "coverage": {"executed_lines", "missing_lines", "executed_branches", "missing_branches"},
     "per_test": {nodeid: {"lines": [...], "branches": [[a, b], ...]}}   # with --per-test}

With --per-test, coverage switches its dynamic context to each test's node id while pytest
runs that test (setup, call and teardown), so one run of many candidate files still credits
every line to the exact test that executed it.
"""

import argparse
import json
import os
import re
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
    parser.add_argument("--per-test", action="store_true", help="per-test coverage contexts")
    args = parser.parse_args(argv)

    for path in (args.repo, os.path.join(args.repo, "src")):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)

    import coverage
    import pytest

    include = os.path.abspath(args.include)
    cov = coverage.Coverage(data_file=None, branch=True, include=[include], config_file=False)
    collector = Collector()
    plugins = [collector]
    if args.per_test:
        class PerTestContext:
            @pytest.hookimpl(hookwrapper=True)
            def pytest_runtest_protocol(self, item, nextitem):
                cov.switch_context(item.nodeid)
                yield
                cov.switch_context("")

        plugins.append(PerTestContext())
    start = time.perf_counter()
    cov.start()
    try:
        exit_code = int(pytest.main(
            [args.tests, "-q", "-p", "no:cacheprovider", "--rootdir", args.tests, "--no-header",
             "--continue-on-collection-errors"],  # one broken file must not sink a batch
            plugins=plugins,
        ))
    finally:
        cov.stop()
    duration = time.perf_counter() - start

    overall = _coverage(cov, include, os.path.dirname(os.path.abspath(args.out)))
    result = {
        "exit_code": exit_code,
        "duration_s": round(duration, 3),
        "tests": list(collector.tests.values()),
        "collection_errors": collector.collection_errors,
        "coverage": overall,
        "per_test": _per_test(cov, include, overall) if args.per_test else {},
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


def _per_test(cov, include, overall):
    """Lines and branches executed by each test (keyed by pytest node id)."""
    data = cov.get_data()
    wanted = os.path.normcase(os.path.abspath(include))
    target = next((f for f in data.measured_files()
                   if os.path.normcase(os.path.abspath(f)) == wanted), None)
    if target is None:
        return {}
    branch_set = {tuple(b) for b in overall.get("executed_branches", [])}
    out = {}
    for context in sorted(data.measured_contexts()):
        if not context:
            continue  # import-time lines and anything outside a test
        data.set_query_contexts(["^" + re.escape(context) + "$"])
        arcs = data.arcs(target) or []
        out[context] = {
            "lines": sorted(data.lines(target) or []),
            "branches": sorted([list(a) for a in arcs if tuple(a) in branch_set]),
        }
    data.set_query_contexts(None)
    return out


if __name__ == "__main__":
    sys.exit(main())
