"""AST-aware test-file editing with libcst (keeps the existing file's formatting).

`splice_tests` merges a candidate test module into an existing test file: new imports go
after the existing imports (duplicates dropped), test functions and helpers are
appended, and name clashes are resolved by deterministic renames (`name_2`, `name_3`,
...) that are also applied to references inside the candidate, so a renamed fixture
still matches the test parameters that use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import libcst as cst


@dataclass
class SpliceResult:
    source: str
    added: list[str] = field(default_factory=list)       # final names of added defs
    renamed: dict[str, str] = field(default_factory=dict)  # candidate name -> new name
    skipped_duplicates: int = 0


def splice_tests(existing: str, candidate: str) -> SpliceResult:
    base = cst.parse_module(existing or "")
    cand = cst.parse_module(candidate)

    existing_imports = {_code(s) for s in base.body if _is_import(s)}
    existing_stmts = {_code(s) for s in base.body}
    taken = _defined_names(base)

    renames: dict[str, str] = {}
    for stmt in cand.body:
        name = _def_name(stmt)
        if name and name in taken and _code(stmt) not in existing_stmts:
            renames[name] = _fresh(name, taken | set(renames.values()) | _defined_names(cand))
    if renames:
        cand = cand.visit(_Renamer(renames))

    new_imports, new_body, added, skipped = [], [], [], 0
    for stmt in cand.body:
        code = _code(stmt)
        if _is_import(stmt):
            if code not in existing_imports:
                new_imports.append(stmt)
                existing_imports.add(code)
        elif code in existing_stmts:
            skipped += 1
        else:
            new_body.append(stmt)
            if (name := _def_name(stmt)) is not None:
                added.append(name)

    body = list(base.body)
    insert_at = _import_insertion_index(body)
    body[insert_at:insert_at] = new_imports
    if new_body:
        first = new_body[0]
        new_body[0] = first.with_changes(leading_lines=[cst.EmptyLine(), cst.EmptyLine()]) \
            if body and isinstance(first, (cst.FunctionDef, cst.ClassDef)) else first
    body.extend(new_body)
    result = base.with_changes(body=body)
    return SpliceResult(result.code, added, renames, skipped)


def remove_tests(source: str, names: set[str]) -> str:
    """Drop top-level functions/classes named in `names`."""
    module = cst.parse_module(source)
    body = [s for s in module.body if _def_name(s) not in names]
    return module.with_changes(body=body).code


def split_tests(source: str) -> list[tuple[str, str]]:
    """Split a generated test module into standalone single-test modules.

    Returns (test_name, module_source) pairs; each module keeps the shared preamble
    (imports, helpers, fixtures, constants) plus exactly one `test_*` function or one
    `Test*` class. Raises libcst.ParserSyntaxError on invalid source.
    """
    module = cst.parse_module(source)
    preamble, tests = [], []
    for stmt in module.body:
        name = _def_name(stmt)
        if name and (name.startswith("test") or name.startswith("Test")):
            tests.append((name, stmt))
        else:
            preamble.append(stmt)
    out = []
    for name, stmt in tests:
        body = [*preamble, stmt]
        out.append((name, module.with_changes(body=body).code))
    return out


def list_tests(source: str) -> list[str]:
    """Top-level `test_*` functions and `Test*` classes, in file order."""
    names = []
    for stmt in cst.parse_module(source).body:
        name = _def_name(stmt)
        if name and (name.startswith("test") or name.startswith("Test")):
            names.append(name)
    return names


# -- helpers ------------------------------------------------------------------------

_EMPTY = cst.Module(body=[])


def _code(node: cst.CSTNode) -> str:
    return _EMPTY.code_for_node(node).strip()


def _is_import(stmt: cst.CSTNode) -> bool:
    return isinstance(stmt, cst.SimpleStatementLine) and all(
        isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body
    )


def _def_name(stmt: cst.CSTNode) -> str | None:
    if isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
        return stmt.name.value
    return None


def _defined_names(module: cst.Module) -> set[str]:
    names = set()
    for stmt in module.body:
        if (name := _def_name(stmt)) is not None:
            names.add(name)
        elif isinstance(stmt, cst.SimpleStatementLine):
            for small in stmt.body:
                if isinstance(small, cst.Assign):
                    for target in small.targets:
                        if isinstance(target.target, cst.Name):
                            names.add(target.target.value)
    return names


def _fresh(name: str, taken: set[str]) -> str:
    n = 2
    while f"{name}_{n}" in taken:
        n += 1
    return f"{name}_{n}"


def _import_insertion_index(body: list[cst.CSTNode]) -> int:
    index = 0
    for i, stmt in enumerate(body):
        if _is_import(stmt):
            index = i + 1
        elif i == 0 and _is_docstring(stmt):
            index = 1
    return index


def _is_docstring(stmt: cst.CSTNode) -> bool:
    return (
        isinstance(stmt, cst.SimpleStatementLine)
        and len(stmt.body) == 1
        and isinstance(stmt.body[0], cst.Expr)
        and isinstance(stmt.body[0].value, cst.SimpleString)
    )


class _Renamer(cst.CSTTransformer):
    """Rename bare Name references (defs, calls, parameters) but not attribute names."""

    def __init__(self, renames: dict[str, str]):
        self.renames = renames
        self._attr_names: set[int] = set()

    def visit_Attribute(self, node: cst.Attribute) -> None:
        self._attr_names.add(id(node.attr))

    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.Name:
        if id(original) in self._attr_names:
            return updated
        new = self.renames.get(original.value)
        return updated.with_changes(value=new) if new else updated
