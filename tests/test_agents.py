import pytest

from autocover.agents.preparer import parse_scenarios, slug
from autocover.llm.prompts import stem_of, writer_messages
from autocover.state import Scenario
from autocover.tools.context import build_module_context
from autocover.tools.splicer import split_tests
from tests.test_context import EXAMPLE_DIR


def test_parse_scenarios_sanitizes_dedupes_and_caps():
    reply = """Here you go:
```json
{"scenarios": [
  {"id": "Child Price!", "kind": "happy", "description": "age 5 -> 5"},
  {"id": "child price", "kind": "edge", "description": "duplicate id"},
  {"id": "65", "kind": "weird", "description": "age 65 -> 7"},
  {"kind": "error"},
  {"id": "neg", "kind": "error", "description": "age -1 -> ValueError"}
]}
```"""
    scenarios = parse_scenarios(reply, "ticket_price", limit=5)
    assert [s.id for s in scenarios] == ["child_price", "s_65", "neg"]
    assert [s.kind for s in scenarios] == ["happy", "happy", "error"]
    assert parse_scenarios(reply, "ticket_price", limit=1)[0].id == "child_price"


def test_parse_scenarios_accepts_bare_list_and_rejects_empty():
    assert parse_scenarios('[{"id": "a", "description": "x"}]', "f", 3)[0].id == "a"
    with pytest.raises(ValueError):
        parse_scenarios('{"scenarios": []}', "f", 3)


def test_slug():
    assert slug("Age == 12 (boundary)") == "age_12_boundary"
    assert slug("") == ""


def test_split_tests_gives_standalone_modules_with_shared_preamble():
    source = (
        "import pytest\nfrom m import f\n\nLIMIT = 3\n\n@pytest.fixture\ndef data():\n"
        "    return [1]\n\ndef test_a(data):\n    assert f(data)\n\n"
        "class TestGroup:\n    def test_b(self):\n        assert True\n"
    )
    parts = split_tests(source)
    assert [name for name, _ in parts] == ["test_a", "TestGroup"]
    for _, module in parts:
        assert "import pytest" in module and "LIMIT = 3" in module and "def data" in module
        compile(module, "<t>", "exec")
    assert "class TestGroup" not in parts[0][1] and "def test_a" not in parts[1][1]


def test_writer_prompt_lists_scenarios_uncovered_lines_and_feedback():
    ctx = build_module_context(EXAMPLE_DIR, "ticket_price.py")
    fn = ctx.function("group_total")
    msgs = writer_messages(
        ctx, fn, [Scenario(id="discount", function="group_total", description="5 -> 45.0")],
        [(19, "total = total * (1 - discount)")], feedback="- test_x: test failed")
    user = msgs[-1]["content"]
    assert "test_group_total__discount" in user
    assert "19: total = total * (1 - discount)" in user
    assert "test_x: test failed" in user
    assert "TEST_WRITER" in msgs[0]["content"]


def test_test_stem():
    assert stem_of("Cart.total") == "Cart_total"


def test_lint_fix_handles_non_ascii():
    from autocover.graph import lint_fix

    src = ('import os\nfrom m import f\n\n\ndef test_a():\n'
           '    """Non‑breaking – café."""\n    assert f()\n')
    fixed = lint_fix(src)
    assert "import os" not in fixed and "‑" in fixed and "café" in fixed
