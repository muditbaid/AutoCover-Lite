from autocover.tools.rules import check_test_source, describe, load_rules

MOD = "ticket_price"


def ids(source: str) -> list[str]:
    return sorted(v.rule for v in check_test_source(source, MOD))


def test_registry_loads_all_rules():
    rules = load_rules()
    assert {"literal_oracle", "no_autouse_fixture", "no_patching_module_under_test",
            "has_assertion", "no_sleep", "no_network"} <= set(rules)
    assert rules["unseeded_random"].severity == "warning"


def test_clean_test_has_no_violations():
    src = ("import pytest\nfrom ticket_price import ticket_price\n\n"
           "def test_a():\n    assert ticket_price(5) == 5\n\n"
           "def test_b():\n    with pytest.raises(ValueError):\n        ticket_price(-1)\n")
    assert ids(src) == []


def test_literal_oracle_detects_computed_expectation_with_taint():
    src = ("from ticket_price import group_total, ticket_price\n\n"
           "def test_g():\n    ages = [10, 30]\n"
           "    raw = sum(ticket_price(a) for a in ages)\n"
           "    expected = round(raw * 0.9, 2)\n"
           "    assert group_total(ages, discount=0.1) == expected\n")
    assert ids(src) == ["literal_oracle"]


def test_literal_oracle_via_module_alias_and_ok_cases():
    bad = ("import ticket_price as tp\n\ndef test_x():\n"
           "    assert tp.group_total([1]) == tp.ticket_price(1)\n")
    assert ids(bad) == ["literal_oracle"]
    ok = ("from ticket_price import group_total\n\ndef test_x():\n"
          "    result = group_total([30] * 5, discount=0.1)\n    assert result == 45.0\n")
    assert ids(ok) == []


def test_autouse_and_patching_module_under_test():
    src = ("import pytest\nimport ticket_price\nfrom ticket_price import group_total\n\n"
           "@pytest.fixture(autouse=True)\ndef fake(monkeypatch):\n"
           "    monkeypatch.setattr('ticket_price.ticket_price', lambda a: 1)\n"
           "    monkeypatch.setattr(ticket_price, 'ticket_price', lambda a: 1)\n\n"
           "def test_x():\n    assert group_total([1]) == 1\n")
    assert ids(src) == ["no_autouse_fixture", "no_patching_module_under_test",
                        "no_patching_module_under_test"]


def test_patching_external_dependency_is_fine():
    src = ("from ticket_price import ticket_price\n\ndef test_x(monkeypatch):\n"
           "    monkeypatch.setattr('os.getcwd', lambda: '/')\n    assert ticket_price(5) == 5\n")
    assert ids(src) == []


def test_missing_assertion_sleep_network_private_random():
    src = ("import time, random, requests\nfrom time import sleep\n"
           "from ticket_price import ticket_price\n\n"
           "def test_x():\n    ticket_price(5)\n    time.sleep(1)\n    sleep(1)\n\n"
           "def test_y(obj):\n    random.randint(0, 3)\n    assert obj._items == []\n")
    assert ids(src) == ["has_assertion", "no_network", "no_sleep", "no_sleep",
                        "private_attribute_assertion", "unseeded_random"]


def test_describe_explains_errors_only():
    src = ("import pytest\n\n@pytest.fixture(autouse=True)\ndef f():\n    pass\n\n"
           "def test_x():\n    assert 1\n")
    violations = check_test_source(src, MOD)
    text = describe(violations)
    assert "[no_autouse_fixture]" in text and "Fix:" in text
    assert describe([v for v in violations if not v.is_error]) == ""
