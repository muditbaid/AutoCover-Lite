"""AST code-context retriever.

Feeds the LLM only what it needs to test one function: how to import it, the module's
imports, the signatures (not bodies) of module-level helpers it calls, the enclosing
class header for methods, and the function's own source. Keeps prompts inside small
free-tier context windows (Cerebras caps at 8K tokens).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FunctionInfo:
    qualname: str  # "func" or "Class.method"
    name: str
    lineno: int
    end_lineno: int
    source: str
    signature: str
    docstring: str | None
    class_name: str | None = None
    is_async: bool = False
    callees: list[str] = field(default_factory=list)  # module-level names it calls

    @property
    def lines(self) -> range:
        return range(self.lineno, self.end_lineno + 1)


@dataclass
class ModuleContext:
    path: Path
    module_name: str
    source: str
    imports: list[str]
    functions: list[FunctionInfo]
    class_headers: dict[str, str]  # class name -> "class X(Base):" + __init__ signature
    helpers: dict[str, FunctionInfo] = field(default_factory=dict)  # private, not targets

    def function(self, qualname: str) -> FunctionInfo:
        for fn in self.functions:
            if fn.qualname == qualname:
                return fn
        raise KeyError(qualname)

    def reachable_helpers(self, fn: FunctionInfo) -> list[FunctionInfo]:
        """Private helpers `fn` calls, transitively, in call-discovery order."""
        found: list[FunctionInfo] = []
        seen, todo = {fn.qualname}, list(fn.callees)
        while todo:
            name = todo.pop(0)
            helper = self.helpers.get(name)
            if helper is None or name in seen:
                continue
            seen.add(name)
            found.append(helper)
            todo.extend(helper.callees)
        return found

    def reach(self, fn: FunctionInfo) -> set[int]:
        """Lines a test of `fn` can reach: its own plus those of the private helpers it
        calls, transitively. Private helpers are not targets, so their gaps count here."""
        lines = set(fn.lines)
        for helper in self.reachable_helpers(fn):
            lines |= set(helper.lines)
        return lines

    def owner_of(self, line: int) -> str | None:
        """Qualname of the (innermost) function or helper containing `line`."""
        best = None
        for fn in [*self.functions, *self.helpers.values()]:
            if fn.lineno <= line <= fn.end_lineno and (best is None or fn.lineno > best.lineno):
                best = fn
        return best.qualname if best else None

    def import_line(self, fn: FunctionInfo) -> str:
        target = fn.class_name or fn.name
        return f"from {self.module_name} import {target}"


def module_name_for(repo_root: str | Path, file: str | Path) -> str:
    """Dotted import path of `file` relative to the repo, honouring a `src/` layout."""
    rel = Path(file).resolve().relative_to(Path(repo_root).resolve())
    parts = list(rel.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_module_context(
    repo_root: str | Path, file: str | Path, *, include_private: bool = False
) -> ModuleContext:
    path = Path(file)
    if not path.is_absolute():
        path = Path(repo_root) / path
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    module_level = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    imports = [ast.get_source_segment(source, n) or ast.unparse(n)
               for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]

    functions: list[FunctionInfo] = []
    class_headers: dict[str, str] = {}

    helpers: dict[str, FunctionInfo] = {}

    def add(node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str | None) -> None:
        # Dunder methods (__init__, __add__, __eq__, ...) are a class's public protocol and
        # often hold most of its logic; only _single and __mangled names are private.
        private = node.name.startswith("_") and not _is_dunder(node.name)
        qualname = f"{class_name}.{node.name}" if class_name else node.name
        # Property setters/deleters and redefinitions share a name: keep them apart.
        accessor = next((d.attr for d in node.decorator_list if isinstance(d, ast.Attribute)
                         and d.attr in ("setter", "deleter")), None)
        if accessor:
            qualname = f"{qualname}_{accessor}"
        taken = {f.qualname for f in functions} | set(helpers)
        n = 2
        base = qualname
        while qualname in taken:
            qualname, n = f"{base}_{n}", n + 1
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        info = FunctionInfo(
            qualname=qualname,
            name=node.name,
            lineno=start,
            end_lineno=node.end_lineno or node.lineno,
            source=_segment(source, start, node.end_lineno or node.lineno),
            signature=signature_of(node),
            docstring=ast.get_docstring(node),
            class_name=class_name,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            callees=sorted((_called_names(node) & (module_level - {node.name}))
                           | {f"{class_name}.{m}" for m in _self_calls(node)} if class_name
                           else _called_names(node) & (module_level - {node.name})),
        )
        if private and not include_private:
            helpers[qualname] = info
        else:
            functions.append(info)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, None)
        elif isinstance(node, ast.ClassDef):
            class_headers[node.name] = _class_header(node)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(item, node.name)

    return ModuleContext(
        path=path,
        module_name=module_name_for(repo_root, path),
        source=source,
        imports=imports,
        functions=functions,
        class_headers=class_headers,
        helpers=helpers,
    )


def render_context(ctx: ModuleContext, qualname: str, *, max_chars: int = 6000) -> str:
    """Prompt-ready context for one function, trimmed to `max_chars`."""
    fn = ctx.function(qualname)
    tree = ast.parse(ctx.source)
    private = ctx.reachable_helpers(fn)
    private_names = {h.qualname for h in private}
    helper_sigs = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in fn.callees and name not in private_names:
            if isinstance(node, ast.ClassDef):
                helper_sigs.append(_class_header(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                first = f'\n    """{doc.splitlines()[0]}"""' if doc else ""
                helper_sigs.append(signature_of(node) + first + "\n    ...")
    header = f"# Module under test: {ctx.module_name}  (import with: {ctx.import_line(fn)})"
    fixed = [header]
    if fn.class_name:
        fixed.append("# Enclosing class:\n" + ctx.class_headers[fn.class_name])
    fixed.append("# Function under test:\n" + fn.source)
    optional = ["# Module imports:\n" + ("\n".join(ctx.imports) or "# (none)")]
    if helper_sigs:
        optional.append("# Other functions it calls (signatures only):\n" +
                        "\n\n".join(helper_sigs))
    if private:
        optional.append("# Private helpers it uses (full source; tests reach them only "
                        "through the function under test):\n" +
                        "\n".join(h.source for h in private))
    # The function itself always fits; optional context is added in order until the
    # budget runs out (the last piece is trimmed).
    budget = max_chars - sum(len(p) + 2 for p in fixed)
    extra = []
    for piece in optional:
        if budget <= 200:
            break
        if len(piece) > budget:
            piece = piece[: budget - 20] + "\n# ...(trimmed)"
        extra.append(piece)
        budget -= len(piece) + 2
    return "\n\n".join([fixed[0], *extra, *fixed[1:]])


def signature_of(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}:"


def _class_header(node: ast.ClassDef) -> str:
    bases = ", ".join(ast.unparse(b) for b in node.bases + node.keywords)
    header = f"class {node.name}({bases}):" if bases else f"class {node.name}:"
    for item in node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            return header + "\n    " + signature_of(item) + "\n        ..."
    return header + "\n    ..."


def _called_names(node: ast.AST) -> set[str]:
    return {
        call.func.id
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    }


def _is_dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _self_calls(node: ast.AST) -> set[str]:
    """Method names called as `self.x(...)` / `cls.x(...)` inside a method."""
    return {
        call.func.attr
        for call in ast.walk(node)
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name) and call.func.value.id in ("self", "cls")
    }


def _segment(source: str, start: int, end: int) -> str:
    return "\n".join(source.splitlines()[start - 1 : end]) + "\n"
