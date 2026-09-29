"""AST-aware test-file editing with libcst (keeps the existing file's formatting).

`splice_tests` merges a candidate test module into an existing test file: new imports go
after the existing imports (duplicates dropped), test functions and helpers are
appended, and name clashes are resolved by deterministic renames (`name_2`, `name_3`,
...) that are also applied to references inside the candidate, so a renamed fixture
still matches the test parameters that use it.

Imports that bind an existing name to something else (`import datetime` in a candidate,
`from datetime import datetime` in the file) get an alias in the candidate, again with
its references renamed: otherwise the later import rebinds the name for every test.
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

    bound = set().union(*(_bindings(s) for s in base.body if _is_import(s)))
    existing_stmts = {_key(s) for s in base.body}
    taken = _defined_names(base)

    # Imports first: a name the file already binds to something else gets an alias.
    import_renames = _import_renames(base, cand, taken)
    if import_renames:
        cand = cand.visit(_Renamer(import_renames)).visit(_AliasImports(import_renames))

    # Rename clashing definitions until none are left: renaming a helper changes the text
    # of the fixtures/tests that use it, which can create new clashes (fixpoint).
    renames: dict[str, str] = {}
    original = cand
    while True:
        current = original.visit(_Renamer(renames)) if renames else original
        new_clashes = {
            name for stmt in current.body
            if (name := _def_name(stmt)) in taken and name not in renames.values()
            and _key(stmt) not in existing_stmts
        }
        if not new_clashes:
            cand = current
            break
        for name in sorted(new_clashes):
            renames[name] = _fresh(name, taken | set(renames.values()) | _defined_names(original))

    new_imports, new_body, added, skipped = [], [], [], 0
    for stmt in cand.body:
        if _is_import(stmt):
            names = _bindings(stmt)
            if not names <= bound:  # skip imports whose every name is already bound
                new_imports.append(stmt)
                bound |= names
        elif _key(stmt) in existing_stmts:
            skipped += 1
        else:
            new_body.append(stmt)
            if (name := _def_name(stmt)) is not None:
                added.append(name)

    body = list(base.body)
    insert_at = _import_insertion_index(body)
    body[insert_at:insert_at] = new_imports
    after_imports = insert_at + len(new_imports)
    if new_imports and after_imports < len(body):
        body[after_imports] = _with_blank_lines(body[after_imports])
    if new_body and body:
        new_body[0] = _with_blank_lines(new_body[0])
    body.extend(new_body)
    result = base.with_changes(body=body)
    return SpliceResult(result.code, added, {**import_renames, **renames}, skipped)


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


def _key(stmt: cst.CSTNode) -> str:
    """Comparison key for duplicate detection: the statement without the blank lines and
    comments above it (splicing re-spaces statements, which must not defeat dedup)."""
    if hasattr(stmt, "leading_lines"):
        stmt = stmt.with_changes(leading_lines=[])
    return _code(stmt)


def _with_blank_lines(stmt: cst.CSTNode, count: int = 2) -> cst.CSTNode:
    """PEP 8 spacing before a top-level def/class, keeping any comments above it."""
    if not isinstance(stmt, (cst.FunctionDef, cst.ClassDef)):
        return stmt
    comments = [line for line in stmt.leading_lines if line.comment is not None]
    return stmt.with_changes(leading_lines=[cst.EmptyLine()] * count + comments)


def _is_import(stmt: cst.CSTNode) -> bool:
    return isinstance(stmt, cst.SimpleStatementLine) and all(
        isinstance(s, (cst.Import, cst.ImportFrom)) for s in stmt.body
    )


def _bindings(stmt: cst.SimpleStatementLine) -> set[tuple]:
    """What an import statement binds, e.g. ("from", "m", "f", None) for `from m import f`."""
    out: set[tuple] = set()
    for small in stmt.body:
        if isinstance(small, cst.Import):
            for alias in small.names:
                asname = _code(alias.asname.name) if alias.asname else None
                out.add(("import", _code(alias.name), asname))
        elif isinstance(small, cst.ImportFrom):
            module = "." * len(small.relative) + (_code(small.module) if small.module else "")
            if isinstance(small.names, cst.ImportStar):
                out.add(("from", module, "*", None))
                continue
            for alias in small.names:
                asname = _code(alias.asname.name) if alias.asname else None
                out.add(("from", module, _code(alias.name), asname))
    return out


def _local(binding: tuple) -> str | None:
    """The name an import binding creates in the module (None for `import *`)."""
    if binding[0] == "import":
        return binding[2] or binding[1].split(".")[0]
    return None if binding[2] == "*" else binding[3] or binding[2]


def _target(binding: tuple) -> tuple:
    """What the binding refers to, whatever local name it gets."""
    return binding[:2] if binding[0] == "import" else binding[:3]


def _import_renames(base: cst.Module, cand: cst.Module, taken: set[str]) -> dict[str, str]:
    """Local import names of `cand` to rename: those the file binds to something else.
    The new name is the file's own name for the same target when it has one."""
    binding_of: dict[str, tuple] = {}   # file: local name -> binding (the last one wins)
    for stmt in base.body:
        if _is_import(stmt):
            for binding in _bindings(stmt):
                if name := _local(binding):
                    binding_of[name] = binding
    # Only names still bound to their import at the end of the file can be reused.
    local_of = {_target(b): name for name, b in sorted(binding_of.items())
                if name not in taken}
    renames: dict[str, str] = {}
    for stmt in cand.body:
        if not _is_import(stmt):
            continue
        for binding in sorted(_bindings(stmt), key=str):
            name = _local(binding)
            if name is None or name in renames:
                continue
            other = binding_of.get(name)
            if not (name in taken or (other and _target(other) != _target(binding))):
                continue
            if binding[0] == "import" and "." in binding[1] and not binding[2]:
                continue  # `import a.b` binds `a`: no alias keeps `a.b.x` working
            renames[name] = local_of.get(_target(binding)) or _fresh(
                name, taken | set(binding_of) | set(renames.values()))
    return renames


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
    """Rename bare Name references (defs, calls, parameters) but not attribute names or
    the modules and names inside import statements (see _AliasImports)."""

    def __init__(self, renames: dict[str, str]):
        self.renames = renames
        self._attr_names: set[int] = set()

    def visit_Import(self, node: cst.Import) -> bool:
        return False

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        return False

    def visit_Attribute(self, node: cst.Attribute) -> None:
        self._attr_names.add(id(node.attr))

    def leave_Name(self, original: cst.Name, updated: cst.Name) -> cst.Name:
        if id(original) in self._attr_names:
            return updated
        new = self.renames.get(original.value)
        return updated.with_changes(value=new) if new else updated


class _AliasImports(cst.CSTTransformer):
    """Bind renamed import names under their new names: `import m` -> `import m as m_2`."""

    def __init__(self, renames: dict[str, str]):
        self.renames = renames

    def leave_ImportAlias(self, original: cst.ImportAlias,
                          updated: cst.ImportAlias) -> cst.ImportAlias:
        local = _code(updated.asname.name) if updated.asname else \
            _code(updated.name).split(".")[0]
        new = self.renames.get(local)
        if new is None:
            return updated
        return updated.with_changes(asname=cst.AsName(name=cst.Name(new)))
