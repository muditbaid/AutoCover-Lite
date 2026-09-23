import ast
from pathlib import Path

from autocover.tools.mutator import find_sites, generate_mutants

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "ticket_price" / "ticket_price.py"


def load(source: str):
    namespace: dict = {}
    exec(compile(source, "<m>", "exec"), namespace)
    return namespace


def test_every_mutant_compiles_and_differs():
    source = EXAMPLE.read_text()
    mutants = generate_mutants(source)
    assert len(mutants) >= 10
    original = ast.unparse(ast.parse(source))
    for m in mutants:
        compile(m.source, "<m>", "exec")
        assert m.source != original
    assert len({m.id for m in mutants}) == len(mutants)


def test_mutants_are_attributed_to_functions():
    mutants = generate_mutants(EXAMPLE.read_text())
    assert {m.function for m in mutants} == {"ticket_price", "group_total"}


def test_docstrings_and_fstrings_are_not_mutated():
    source = '"""Module doc."""\n\ndef f(x):\n    """Doc."""\n    return f"v={x}"\n'
    kinds = [(s.kind, s.description) for s in find_sites(source)]
    assert kinds == [("return", "return value -> None")]


def test_boundary_mutant_changes_behaviour():
    source = "def adult(age):\n    return age >= 18\n"
    (mutant,) = [m for m in generate_mutants(source) if m.kind == "compare"]
    assert mutant.description == ">= -> >"
    assert load(source)["adult"](18) is True
    assert load(mutant.source)["adult"](18) is False  # a test with age=18 kills it


def test_operator_coverage():
    source = (
        "def f(a, b, flag):\n"
        "    total = a + b\n"
        "    total *= 2\n"
        "    if not flag and a < b:\n"
        "        return total\n"
        "    return -1\n"
    )
    kinds = {s.kind for s in find_sites(source)}
    assert {"binop", "augassign", "not", "boolop", "compare", "constant", "return"} <= kinds


def test_function_and_line_filters():
    source = EXAMPLE.read_text()
    only_tp = generate_mutants(source, functions={"ticket_price"})
    assert only_tp and all(m.function == "ticket_price" for m in only_tp)
    line = only_tp[0].lineno
    assert all(m.lineno == line for m in generate_mutants(source, lines={line}))


def test_cap_is_deterministic_per_seed():
    source = EXAMPLE.read_text()
    a = generate_mutants(source, max_per_function=3, seed=1)
    b = generate_mutants(source, max_per_function=3, seed=1)
    assert [m.id for m in a] == [m.id for m in b]
    for fn in ("ticket_price", "group_total"):
        assert sum(m.function == fn for m in a) <= 3
