"""Bounded AST mutation testing.

A mutant is the module with one small, deliberate bug: `+` becomes `-`, `<` becomes `<=`,
`and` becomes `or`, a constant is nudged, a return value becomes None, and so on. A good
test fails on the mutant ("kills" it); a test that still passes is too weak.

Mutants are enumerated in a fixed order (ast.walk), attributed to their enclosing
function, and capped per function with a seeded sample, as in the paper's "bounded type
and count of mutants". Mutated modules are produced with ast.unparse, which drops
comments; only behaviour matters for running tests.
"""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass

from autocover.config import MutationConfig

BINOP_SWAP: dict[type, type] = {
    ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Div, ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult, ast.Mod: ast.FloorDiv, ast.Pow: ast.Mult,
}
CMP_SWAP: dict[type, type] = {
    ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
    ast.In: ast.NotIn, ast.NotIn: ast.In,
}
SYMBOL = {
    ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//",
    ast.Mod: "%", ast.Pow: "**", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
    ast.Eq: "==", ast.NotEq: "!=", ast.Is: "is", ast.IsNot: "is not", ast.In: "in",
    ast.NotIn: "not in", ast.And: "and", ast.Or: "or",
}


@dataclass(frozen=True)
class Site:
    node_index: int  # position in ast.walk order
    kind: str        # binop | compare | boolop | not | constant | return | augassign
    detail: int      # which comparator for chained compares, else 0
    lineno: int
    function: str | None
    description: str


@dataclass(frozen=True)
class Mutant:
    id: str
    function: str | None
    lineno: int
    kind: str
    description: str
    source: str


def find_sites(source: str) -> list[Site]:
    tree = ast.parse(source)
    parents = _parent_map(tree)
    skip = _docstring_nodes(tree)
    ranges = _function_ranges(tree)
    sites: list[Site] = []
    for index, node in enumerate(ast.walk(tree)):
        for kind, detail, desc in _mutations_for(node, parents, skip):
            lineno = getattr(node, "lineno", 0)
            sites.append(Site(index, kind, detail, lineno, _enclosing(ranges, lineno), desc))
    return sites


def generate_mutants(
    source: str,
    *,
    functions: set[str] | None = None,
    lines: set[int] | None = None,
    max_per_function: int | None = None,
    seed: int = 0,
) -> list[Mutant]:
    """Mutants of `source`, optionally limited to some functions / executed lines."""
    sites = [
        s for s in find_sites(source)
        if (functions is None or s.function in functions) and (lines is None or s.lineno in lines)
    ]
    if max_per_function is not None:
        rng = random.Random(seed)
        by_fn: dict[str | None, list[Site]] = {}
        for site in sites:
            by_fn.setdefault(site.function, []).append(site)
        sites = []
        for group in by_fn.values():
            if len(group) > max_per_function:
                group = rng.sample(group, max_per_function)
            chosen = group
            sites.extend(sorted(chosen, key=lambda s: (s.node_index, s.detail)))
    original = ast.unparse(ast.parse(source))
    mutants = []
    for site in sites:
        mutated = apply_site(source, site)
        if mutated is not None and mutated != original:
            mutants.append(Mutant(
                id=f"m{site.node_index}_{site.detail}_{site.kind}",
                function=site.function, lineno=site.lineno, kind=site.kind,
                description=site.description, source=mutated,
            ))
    return mutants


def build_mutant_pool(source: str, cfg: MutationConfig) -> list[Mutant]:
    """The module's mutant pool: per-function cap, then a seeded total cap. Deterministic,
    so different tools (e.g. the benchmark baseline) are scored on identical mutants."""
    pool = generate_mutants(source, max_per_function=cfg.max_mutants_per_function,
                            seed=cfg.seed)
    if cfg.max_mutants_total and len(pool) > cfg.max_mutants_total:
        keep = set(random.Random(cfg.seed).sample(range(len(pool)), cfg.max_mutants_total))
        pool = [m for i, m in enumerate(pool) if i in keep]
    return pool


def apply_site(source: str, site: Site) -> str | None:
    tree = ast.parse(source)
    parents = _parent_map(tree)
    node = list(ast.walk(tree))[site.node_index]
    if site.kind in ("binop", "augassign"):
        node.op = BINOP_SWAP[type(node.op)]()
    elif site.kind == "compare":
        ops = list(node.ops)
        ops[site.detail] = CMP_SWAP[type(ops[site.detail])]()
        node.ops = ops
    elif site.kind == "boolop":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif site.kind == "not":
        _replace(parents[node], node, node.operand)
    elif site.kind == "constant":
        node.value = _nudge(node.value)
    elif site.kind == "return":
        node.value = ast.Constant(value=None)
    else:  # pragma: no cover - defensive
        return None
    tree = ast.fix_missing_locations(tree)
    try:
        mutated = ast.unparse(tree)
        compile(mutated, "<mutant>", "exec")
    except (SyntaxError, ValueError):
        return None
    return mutated


# -- helpers ------------------------------------------------------------------------


def _mutations_for(node, parents, skip):
    if isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAP:
        new = BINOP_SWAP[type(node.op)]
        yield "binop", 0, f"{SYMBOL[type(node.op)]} -> {SYMBOL[new]}"
    elif isinstance(node, ast.AugAssign) and type(node.op) in BINOP_SWAP:
        new = BINOP_SWAP[type(node.op)]
        yield "augassign", 0, f"{SYMBOL[type(node.op)]}= -> {SYMBOL[new]}="
    elif isinstance(node, ast.Compare):
        for i, op in enumerate(node.ops):
            if type(op) in CMP_SWAP:
                yield "compare", i, f"{SYMBOL[type(op)]} -> {SYMBOL[CMP_SWAP[type(op)]]}"
    elif isinstance(node, ast.BoolOp):
        other = "or" if isinstance(node.op, ast.And) else "and"
        yield "boolop", 0, f"{SYMBOL[type(node.op)]} -> {other}"
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        yield "not", 0, "remove `not`"
    elif isinstance(node, ast.Constant) and id(node) not in skip:
        parent = parents.get(node)
        if isinstance(parent, (ast.JoinedStr, ast.FormattedValue)):
            return
        if isinstance(parent, ast.Return) and node.value is None:
            return
        nudged = _nudge(node.value)
        if nudged is not _NO_CHANGE:
            yield "constant", 0, f"{node.value!r} -> {nudged!r}"
    elif isinstance(node, ast.Return) and node.value is not None:
        if not (isinstance(node.value, ast.Constant) and node.value.value is None):
            yield "return", 0, "return value -> None"


_NO_CHANGE = object()


def _nudge(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return "" if value else "XX"
    return _NO_CHANGE


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                ids.add(id(body[0].value))
    return ids


def _function_ranges(tree: ast.Module) -> list[tuple[int, int, str]]:
    ranges = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ranges.append((node.lineno, node.end_lineno, node.name))
        elif isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ranges.append((item.lineno, item.end_lineno, f"{node.name}.{item.name}"))
    return ranges


def _enclosing(ranges, lineno: int) -> str | None:
    best = None
    for start, end, name in ranges:
        if start <= lineno <= end and (best is None or start >= best[0]):
            best = (start, name)
    return best[1] if best else None


def _replace(parent: ast.AST, old: ast.AST, new: ast.AST) -> None:
    for field, value in ast.iter_fields(parent):
        if value is old:
            setattr(parent, field, new)
            return
        if isinstance(value, list):
            for i, item in enumerate(value):
                if item is old:
                    value[i] = new
                    return
    raise ValueError("node not found in parent")  # pragma: no cover

