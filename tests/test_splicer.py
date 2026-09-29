from autocover.tools.splicer import list_tests, remove_tests, splice_tests

EXISTING = '''"""Tests for ticket_price."""
import pytest

from ticket_price import ticket_price


def test_child():  # keep this comment
    assert ticket_price(5) == 5
'''


def test_adds_new_imports_after_existing_and_dedupes():
    candidate = "import pytest\nimport math\n\ndef test_adult():\n    assert True\n"
    result = splice_tests(EXISTING, candidate)
    src = result.source
    assert src.count("import pytest") == 1
    assert src.index("import math") < src.index("def test_child")
    assert "# keep this comment" in src  # formatting preserved
    assert result.added == ["test_adult"]
    assert list_tests(src) == ["test_child", "test_adult"]


def test_clashing_test_name_is_renamed():
    candidate = "def test_child():\n    assert ticket_price(11) == 5\n"
    result = splice_tests(EXISTING, candidate)
    assert result.renamed == {"test_child": "test_child_2"}
    assert list_tests(result.source) == ["test_child", "test_child_2"]


def test_identical_test_is_skipped():
    candidate = "def test_child():  # keep this comment\n    assert ticket_price(5) == 5\n"
    result = splice_tests(EXISTING, candidate)
    assert result.added == [] and result.skipped_duplicates == 1


def test_renamed_fixture_updates_params_but_not_attributes():
    existing = "import pytest\n\n@pytest.fixture\ndef visitor():\n    return 1\n"
    candidate = (
        "import pytest\n\n@pytest.fixture\ndef visitor():\n    return 2\n\n"
        "def test_x(visitor):\n    obj = object()\n    getattr(obj, 'visitor', None)\n"
        "    assert visitor == 2 and not hasattr(obj.__class__, 'visitor')\n"
        "    ns = type('N', (), {})()\n    ns.visitor = visitor\n"
    )
    src = splice_tests(existing, candidate).source
    assert "def visitor_2():" in src
    assert "def test_x(visitor_2):" in src
    assert "ns.visitor = visitor_2" in src  # attribute name kept, reference renamed


def test_splice_into_empty_file():
    result = splice_tests("", "import pytest\n\ndef test_a():\n    assert 1\n")
    assert result.source.startswith("import pytest")
    assert list_tests(result.source) == ["test_a"]


def test_remove_tests():
    src = splice_tests(EXISTING, "def test_adult():\n    assert True\n").source
    assert list_tests(remove_tests(src, {"test_child"})) == ["test_adult"]


def test_imports_already_bound_by_a_broader_import_are_skipped():
    existing = "from ticket_price import group_total, ticket_price\n\ndef test_a():\n    pass\n"
    candidate = (
        "from ticket_price import ticket_price\n"
        "import pytest\n"
        "from ticket_price import ticket_price as tp\n\n"
        "def test_b():\n    pass\n"
    )
    src = splice_tests(existing, candidate).source
    assert src.count("from ticket_price import ticket_price\n") == 0
    assert "import pytest" in src
    assert "from ticket_price import ticket_price as tp" in src  # new binding: kept


HELPER_PREAMBLE = '''import pytest
from ticket_price import group_total

# helper used to isolate group_total
def _fake(age):
    return age // 10

@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr("ticket_price.ticket_price", _fake)

'''


def test_identical_preamble_with_comments_is_not_duplicated():
    first = splice_tests("def test_t1():\n    assert True\n",
                         HELPER_PREAMBLE + "def test_a(patched):\n    assert 1\n").source
    result = splice_tests(first, HELPER_PREAMBLE + "def test_b(patched):\n    assert 2\n")
    assert result.renamed == {} and result.skipped_duplicates == 2
    assert result.source.count("def _fake(") == 1 and result.source.count("def patched(") == 1
    assert "# helper used to isolate group_total" in result.source  # comment survives


def test_renaming_a_helper_cascades_to_fixtures_that_use_it():
    first = splice_tests("", HELPER_PREAMBLE + "def test_a(patched):\n    assert 1\n").source
    changed = HELPER_PREAMBLE.replace("age // 10", "age // 20")
    result = splice_tests(first, changed + "def test_b(patched):\n    assert 2\n")
    assert result.renamed == {"_fake": "_fake_2", "patched": "patched_2"}
    src = result.source
    assert src.count("def patched(") == 1 and "def patched_2(monkeypatch):" in src
    assert '"ticket_price.ticket_price", _fake_2)' in src
    assert "def test_b(patched_2):" in src
    compile(src, "<suite>", "exec")


def _run_all(source: str) -> list[str]:
    """Execute a merged test module and call every test; return the names that passed."""
    namespace: dict = {}
    exec(compile(source, "<suite>", "exec"), namespace)
    passed = []
    for name in list_tests(source):
        namespace[name]()
        passed.append(name)
    return passed


def test_import_binding_a_name_differently_gets_an_alias():
    # dateutil benchmark: `import datetime` tests merged after `from datetime import
    # datetime` tests; the later import rebound `datetime` and 12 tests failed together.
    suite = "from datetime import datetime\n\n\ndef test_cls():\n" \
            "    assert datetime(2020, 1, 1).year == 2020\n"
    module_style = "import datetime\n\n\ndef test_mod():\n" \
                   "    assert datetime.datetime(2021, 1, 1).year == 2021\n" \
                   "    assert datetime.date(2021, 1, 2).day == 2\n"
    merged = splice_tests(suite, module_style)
    assert merged.renamed == {"datetime": "datetime_2"}
    assert "import datetime as datetime_2" in merged.source
    assert "datetime_2.datetime(2021, 1, 1)" in merged.source
    # A later candidate importing the module again reuses the alias, no new import.
    again = splice_tests(merged.source, module_style.replace("test_mod", "test_mod_b"))
    assert again.source.count("import datetime as datetime_2") == 1
    assert "import datetime" not in again.source.splitlines()
    assert _run_all(again.source) == ["test_cls", "test_mod", "test_mod_b"]


def test_import_clashing_with_a_helper_name_gets_an_alias():
    suite = "def json():\n    return 'helper'\n\n\ndef test_helper():\n" \
            "    assert json() == 'helper'\n"
    cand = "import json\n\n\ndef test_dump():\n    assert json.dumps([1]) == '[1]'\n"
    merged = splice_tests(suite, cand)
    assert "import json as json_2" in merged.source
    assert _run_all(merged.source) == ["test_helper", "test_dump"]


def test_same_import_in_another_form_is_not_renamed():
    suite = "from os import path\n\n\ndef test_a():\n    assert path.sep\n"
    cand = "from os import path\n\n\ndef test_b():\n    assert path.join('a', 'b')\n"
    merged = splice_tests(suite, cand)
    assert merged.renamed == {} and merged.source.count("from os import path") == 1
