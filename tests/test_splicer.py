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
