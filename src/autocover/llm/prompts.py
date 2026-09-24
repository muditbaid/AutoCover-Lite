"""Prompt templates.

Each prompt is split the way the AutoCover paper does it (section 5.2), so provider-side
prompt caching and our literal cache get the most reuse:

* stable      - role, rules and output format: identical for every call (system message)
* semi-stable - the function's code context: changes only when the source changes
* volatile    - scenarios, uncovered lines, previous failures: changes every round
"""

from __future__ import annotations

from autocover.state import Scenario
from autocover.tools.context import FunctionInfo, ModuleContext, render_context

TEST_RULES = """\
- Use pytest. Plain functions, plain `assert`; `pytest.raises` for expected exceptions;
  `pytest.mark.parametrize` is fine for several inputs of ONE scenario.
- Assert observable behaviour (return values, raised exceptions, state reachable through
  the public API). Never assert on private attributes or implementation details.
- Deterministic and hermetic: no network, no sleeping, no real clock or randomness without
  a fixed seed, no files outside pytest's `tmp_path`, no reliance on test order.
- Import the code under test exactly as shown in the context. Do not redefine or copy the
  function under test, and do not mock it. Mock only true external dependencies.
- Every expected value must follow from the code's actual logic - trace it carefully,
  especially at boundaries. A wrong expectation makes the test fail and it is discarded.
- Keep each test short and focused; no print statements, no commented-out code."""

PLANNER_SYSTEM = f"""\
You are the Preparer agent of an automated unit-test generator (SCENARIO_PLANNER).
Given one Python function, list the distinct behaviours ("scenarios") a thorough test
suite must check: normal cases, boundaries (exact thresholds, empty inputs, zero, one),
and error paths (raised exceptions). Each scenario must be checkable through the public
API and must differ in behaviour, not just in input values.

Reply with JSON only, in exactly this shape:
{{"scenarios": [{{"id": "short_snake_case_id", "kind": "happy|edge|error",
                 "description": "input -> expected observable behaviour"}}]}}

Rules for tests that will later be written from your scenarios:
{TEST_RULES}"""

WRITER_SYSTEM = f"""\
You are the Generator agent of an automated unit-test generator (TEST_WRITER).
Write pytest tests for ONE Python function, one test function per requested scenario.

Rules:
{TEST_RULES}

Output format:
- Reply with a single ```python code block containing a complete test module: imports,
  any small helpers or fixtures, then the tests. Nothing outside the code block.
- Name each test exactly `test_<function>__<scenario_id>` using the ids given. Extra tests
  that cover listed uncovered lines are named `test_<function>__cover_<n>`."""


def planner_messages(ctx: ModuleContext, fn: FunctionInfo, max_scenarios: int) -> list[dict]:
    user = (
        f"{render_context(ctx, fn.qualname)}\n\n"
        f"List at most {max_scenarios} scenarios for `{fn.qualname}`."
    )
    return [{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": user}]


def writer_messages(
    ctx: ModuleContext,
    fn: FunctionInfo,
    scenarios: list[Scenario],
    uncovered: list[tuple[int, str]],
    feedback: str = "",
) -> list[dict]:
    test_prefix = f"test_{stem_of(fn.qualname)}__"
    parts = [render_context(ctx, fn.qualname)]
    if scenarios:
        listed = "\n".join(f"- {test_prefix}{s.id}  [{s.kind}] {s.description}"
                           for s in scenarios)
        parts.append(f"Write one test per scenario below:\n{listed}")
    if uncovered:
        lines = "\n".join(f"  {n:>4}: {text}" for n, text in uncovered)
        parts.append("These lines of the function are still not executed by any test. "
                     "Make sure some test reaches each of them:\n" + lines)
    if feedback:
        parts.append("Previous attempts for this function failed like this - do not repeat "
                     "these mistakes:\n" + feedback)
    return [{"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


def stem_of(qualname: str) -> str:
    """`Class.method` -> `Class_method` (valid inside a test function name)."""
    return qualname.replace(".", "_")
