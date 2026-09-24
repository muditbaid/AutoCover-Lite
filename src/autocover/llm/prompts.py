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
  function under test, and do not mock or monkeypatch it or other code of the same module
  (call its real helpers). Mock only true external dependencies (network, files, clock).
- No `autouse=True` fixtures and no module-level state changes: every test is later merged
  into one shared test file and must not affect the other tests there.
- Every expected value must follow from the code's actual logic - trace it carefully,
  especially at boundaries. A wrong expectation makes the test fail and it is discarded.
- Write expected values as literals (e.g. `== 45.0`). Never compute them by calling the
  code under test or its helpers: such a test still passes when that code is broken.
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


PLANNER_GROUP_SYSTEM = PLANNER_SYSTEM.replace(
    "Given one Python function, list", "Given several Python functions, list for each one"
).replace(
    '{"scenarios": [{"id": "short_snake_case_id", "kind": "happy|edge|error",\n'
    '                 "description": "input -> expected observable behaviour"}]}',
    '{"functions": {"<function name exactly as given>": [{"id": "short_snake_case_id",\n'
    '   "kind": "happy|edge|error", "description": "input -> expected observable behaviour"}]}}',
)


WRITER_CONTEXT_CHARS = 12_000  # room for private helper sources the tests must reach


def planner_group_messages(ctx: ModuleContext, fns: list[FunctionInfo], max_scenarios: int,
                           max_chars_each: int = 3500) -> list[dict]:
    """One planning call for several functions (cheaper on large modules)."""
    blocks = [f"=== Function `{fn.qualname}` ===\n"
              f"{render_context(ctx, fn.qualname, max_chars=max_chars_each)}" for fn in fns]
    names = ", ".join(f"`{fn.qualname}`" for fn in fns)
    user = ("\n\n".join(blocks) +
            f"\n\nList at most {max_scenarios} scenarios for each of: {names}.")
    return [{"role": "system", "content": PLANNER_GROUP_SYSTEM},
            {"role": "user", "content": user}]


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
    survivors: list[tuple[int, str, str]] | None = None,
) -> list[dict]:
    test_prefix = f"test_{stem_of(fn.qualname)}__"
    parts = [render_context(ctx, fn.qualname, max_chars=WRITER_CONTEXT_CHARS)]
    if scenarios:
        listed = "\n".join(f"- {test_prefix}{s.id}  [{s.kind}] {s.description}"
                           for s in scenarios)
        parts.append(f"Write one test per scenario below:\n{listed}")
    if uncovered:
        lines = "\n".join(f"  {n:>4}: {text}" for n, text in uncovered)
        parts.append("These lines of the function are still not executed by any test. "
                     "Make sure some test reaches each of them:\n" + lines)
    if survivors:
        listed = "\n".join(f"  {n:>4}: `{code}` with {change}" for n, code, change in survivors)
        parts.append("Every existing test still passes when the code is deliberately broken "
                     "in these ways. Write tests (literal expected values, exact boundaries) "
                     "that would FAIL on each of these bugs:\n" + listed)
    if feedback:
        parts.append("Previous attempts for this function failed like this - do not repeat "
                     "these mistakes:\n" + feedback)
    return [{"role": "system", "content": WRITER_SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]


FIXER_SYSTEM = f"""\
You are the Fixer agent of an automated unit-test generator (TEST_FIXER).
You get ONE pytest test that failed or was rejected, and why. Return a corrected version
with the smallest change that fixes the problem, keeping the same intent and test name.
If the test's expectation is wrong, re-derive the correct value from the code under test;
if the code under test is behaving correctly, never change the test to hide it by
weakening assertions.

Rules:
{TEST_RULES}

Reply with a single ```python code block containing the complete test module (imports,
helpers, the one test). Nothing outside the code block."""

JUDGE_SYSTEM = """\
You are the scenario judge of an automated unit-test generator (SCENARIO_JUDGE).
Decide whether a test really checks the stated scenario: it must drive the code with
inputs matching the scenario and assert the scenario's expected behaviour (a return
value, an exception, or observable state). A test that only runs the code, asserts
something unrelated, or checks a different situation does NOT cover the scenario.
Reply with JSON only: {"covers": true|false, "reason": "one short sentence"}"""


def fixer_messages(ctx: ModuleContext, fn: FunctionInfo, test_code: str, test_name: str,
                   problem: str) -> list[dict]:
    user = (f"{render_context(ctx, fn.qualname, max_chars=WRITER_CONTEXT_CHARS)}\n\n"
            f"Test `{test_name}`:\n```python\n{test_code.rstrip()}\n```\n\n"
            f"Problem:\n{problem.strip()}\n\n"
            f"Return the fixed test, still named `{test_name}`.")
    return [{"role": "system", "content": FIXER_SYSTEM}, {"role": "user", "content": user}]


def judge_messages(fn: FunctionInfo, scenario: Scenario, test_code: str) -> list[dict]:
    user = (f"Function under test:\n```python\n{fn.source.rstrip()}\n```\n\n"
            f"Scenario [{scenario.kind}]: {scenario.description}\n\n"
            f"Test:\n```python\n{test_code.rstrip()}\n```\n\nDoes the test cover the scenario?")
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]


def stem_of(qualname: str) -> str:
    """`Class.method` -> `Class_method` (valid inside a test function name)."""
    return qualname.replace(".", "_")
