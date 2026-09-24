"""Deterministic best-practice checks for candidate tests (the Validator's rule gate).

Rule metadata (severity, rationale, fix text, example) lives in
`autocover/rules/best_practices.yaml`; the checks themselves are AST passes here.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml

REGISTRY = Path(__file__).resolve().parents[1] / "rules" / "best_practices.yaml"
NETWORK_MODULES = {"socket", "requests", "httpx", "aiohttp", "urllib.request", "http.client",
                   "urllib3", "websocket", "websockets"}
PATCH_CALLS = {"setattr", "patch", "object", "setitem"}
RANDOM_CALLS = {"random", "randint", "randrange", "choice", "choices", "shuffle", "sample",
                "uniform", "gauss"}


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    rationale: str
    fix: str
    example: str


@dataclass(frozen=True)
class Violation:
    rule: str
    severity: str
    line: int
    message: str

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


@cache
def load_rules(path: str | None = None) -> dict[str, Rule]:
    data = yaml.safe_load(Path(path or REGISTRY).read_text(encoding="utf-8"))
    return {r["id"]: Rule(r["id"], r["severity"], r["rationale"].strip(), r["fix"].strip(),
                          r.get("example", "").rstrip()) for r in data["rules"]}


def check_test_source(source: str, module_name: str) -> list[Violation]:
    """Violations of the registry rules in one candidate test module."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # the sandbox run reports syntax errors with better diagnostics
    rules = load_rules()
    names, aliases = _module_bindings(tree, module_name)
    found: list[tuple[str, int, str]] = []
    found += _network(tree)
    found += _autouse(tree)
    found += _patching(tree, module_name, aliases)
    found += _sleep(tree)
    found += _random(tree)
    found += _introspection(tree)
    for fn in _test_functions(tree):
        found += _has_assertion(fn)
        found += _literal_oracle(fn, names, aliases)
        found += _private_attrs(fn)
    return [Violation(rid, rules[rid].severity, line, msg) for rid, line, msg in found
            if rid in rules]


def describe(violations: list[Violation]) -> str:
    """Fixer-facing explanation: what is wrong, why, and how to fix it."""
    rules = load_rules()
    parts = []
    for rid in dict.fromkeys(v.rule for v in violations if v.is_error):
        rule = rules[rid]
        lines = ", ".join(str(v.line) for v in violations if v.rule == rid)
        parts.append(f"- [{rid}] line(s) {lines}: {rule.rationale}\n  Fix: {rule.fix}\n"
                     + "\n".join("    " + x for x in rule.example.splitlines()))
    return "\n".join(parts)


# -- checks --------------------------------------------------------------------------


def _module_bindings(tree: ast.Module, module_name: str) -> tuple[set[str], set[str]]:
    """Local names bound to things from the module under test, and aliases of the module."""
    names, aliases = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            names |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == module_name:
                    aliases.add(a.asname or a.name.split(".")[0])
    return names, aliases


def _test_functions(tree: ast.Module):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name.startswith("test"):
            yield node


def _calls_module(node: ast.AST, names: set[str], aliases: set[str],
                  tainted: set[str] = frozenset()) -> bool:
    """Does `node` call code under test (directly, via the module alias, or via a variable
    whose value was computed that way)?"""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            func = sub.func
            if isinstance(func, ast.Name) and func.id in names:
                return True
            if isinstance(func, ast.Attribute):
                root = func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name) and (root.id in aliases or root.id in names):
                    return True
        elif isinstance(sub, ast.Name) and sub.id in tainted:
            return True
    return False


def _literal_oracle(fn, names: set[str], aliases: set[str]):
    """`assert f(x) == <something also computed with the module>` -> both sides move
    together when the code under test is broken."""
    tainted: set[str] = set()
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.Assign) and _calls_module(stmt.value, names, aliases, tainted):
            tainted |= {t.id for t in stmt.targets if isinstance(t, ast.Name)}
        elif isinstance(stmt, ast.Assert) and isinstance(stmt.test, ast.Compare):
            cmp = stmt.test
            if not any(isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)) for op in cmp.ops):
                continue
            sides = [cmp.left, *cmp.comparators]
            derived = [s for s in sides if _calls_module(s, names, aliases, tainted)]
            if len(derived) >= 2:
                yield ("literal_oracle", stmt.lineno,
                       "expected value is computed with the code under test")


def _has_assertion(fn):
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            return
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in ("raises", "warns", "deprecated_call", "fail") or \
                    name.startswith("assert"):
                return
    yield ("has_assertion", fn.lineno, f"{fn.name} contains no assertion")


def _autouse(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                if isinstance(dec, ast.Call) and any(
                        k.arg == "autouse" and isinstance(k.value, ast.Constant) and k.value.value
                        for k in dec.keywords):
                    yield ("no_autouse_fixture", dec.lineno, f"fixture {node.name} is autouse")


def _patching(tree, module_name: str, aliases: set[str]):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if name not in PATCH_CALLS:
            continue
        target = node.args[0]
        hits = (isinstance(target, ast.Constant) and isinstance(target.value, str)
                and (target.value == module_name or target.value.startswith(module_name + ".")))
        hits |= isinstance(target, ast.Name) and target.id in aliases
        if hits:
            yield ("no_patching_module_under_test", node.lineno,
                   "patches code of the module under test")


def _sleep(tree):
    sleep_names = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                   and n.module == "time" for a in n.names if a.name == "sleep"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if (isinstance(func, ast.Attribute) and func.attr == "sleep"
                    and isinstance(func.value, ast.Name) and func.value.id == "time") or \
                    (isinstance(func, ast.Name) and func.id in sleep_names):
                yield ("no_sleep", node.lineno, "time.sleep in a test")


def _network(tree):
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for mod in mods:
            if mod in NETWORK_MODULES or mod.split(".")[0] in {"requests", "httpx", "aiohttp"}:
                yield ("no_network", node.lineno, f"imports network module {mod}")


def _private_attrs(fn):
    for node in ast.walk(fn):
        if isinstance(node, ast.Assert):
            for sub in ast.walk(node.test):
                if isinstance(sub, ast.Attribute) and sub.attr.startswith("_") and \
                        not sub.attr.startswith("__"):
                    yield ("private_attribute_assertion", node.lineno,
                           f"asserts on private attribute {sub.attr}")
                    break


INTROSPECTION_ATTRS = {"__defaults__", "__kwdefaults__", "__code__", "__closure__",
                       "__annotations__", "__wrapped__"}


def _introspection(tree):
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module]
            if "inspect" in mods:
                yield ("no_introspection", node.lineno, "imports inspect")
        elif isinstance(node, ast.Attribute) and node.attr in INTROSPECTION_ATTRS:
            yield ("no_introspection", node.lineno, f"reads {node.attr}")


def _random(tree):
    seeded = any(isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "seed"
                 for n in ast.walk(tree))
    if seeded:
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and \
                isinstance(node.func.value, ast.Name) and node.func.value.id == "random" and \
                node.func.attr in RANDOM_CALLS:
            yield ("unseeded_random", node.lineno, f"random.{node.func.attr} without a seed")
