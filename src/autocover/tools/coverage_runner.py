"""Accumulated coverage bookkeeping across accepted tests.

AutoCover keeps a test only if it adds signal. `CoverageTracker.gain` answers "which
lines / branches would this run add on top of what the accepted suite already covers?",
and `add` commits a run once its test is accepted.
"""

from __future__ import annotations

from dataclasses import dataclass

from autocover.tools.sandbox import CoverageData


@dataclass(frozen=True)
class Gain:
    lines: frozenset[int]
    branches: frozenset[tuple[int, int]]

    def __bool__(self) -> bool:
        return bool(self.lines or self.branches)


class CoverageTracker:
    def __init__(self, baseline: CoverageData | None = None):
        self.lines: set[int] = set()
        self.branches: set[tuple[int, int]] = set()
        self.universe_lines: set[int] = set()
        self.universe_branches: set[tuple[int, int]] = set()
        if baseline is not None:
            self.add(baseline)

    def observe(self, cov: CoverageData) -> None:
        """Learn the module's measurable lines/branches from any measured run."""
        if cov.measured:
            self.universe_lines |= cov.all_lines
            self.universe_branches |= cov.all_branches

    def gain(self, cov: CoverageData) -> Gain:
        self.observe(cov)
        return Gain(frozenset(cov.executed_lines - self.lines),
                    frozenset(cov.executed_branches - self.branches))

    def add(self, cov: CoverageData) -> Gain:
        gained = self.gain(cov)
        self.lines |= cov.executed_lines
        self.branches |= cov.executed_branches
        return gained

    @property
    def uncovered_lines(self) -> set[int]:
        return self.universe_lines - self.lines

    @property
    def line_pct(self) -> float:
        return _pct(len(self.lines & self.universe_lines), len(self.universe_lines))

    @property
    def branch_pct(self) -> float:
        return _pct(len(self.branches & self.universe_branches), len(self.universe_branches))

    def summary(self) -> dict:
        return {
            "line_pct": round(self.line_pct, 1),
            "branch_pct": round(self.branch_pct, 1),
            "lines_covered": len(self.lines & self.universe_lines),
            "lines_total": len(self.universe_lines),
            "branches_covered": len(self.branches & self.universe_branches),
            "branches_total": len(self.universe_branches),
        }


def _pct(part: int, whole: int) -> float:
    return 100.0 * part / whole if whole else 0.0
