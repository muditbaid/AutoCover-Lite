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

    def function(self, qualname: str) -> FunctionInfo:
        for fn in self.functions:
            if fn.qualname == qualname:
                return fn
        raise KeyError(qualname)

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

    def add(node: ast.FunctionDef | ast.AsyncFunctionDef, class_name: str | None) -> None:
        if node.name.startswith("_") and not include_private and node.name != "__call__":
            return
        qualname = f"{class_name}.{node.name}" if class_name else node.name
        start = node.decorator_list[0].lineno if node.decorator_list else node.lineno
        functions.append(FunctionInfo(
            qualname=qualname,
            name=node.name,
            lineno=start,
            end_lineno=node.end_lineno or node.lineno,
            source=_segment(source, start, node.end_lineno or node.lineno),
            signature=signature_of(node),
            docstring=ast.get_docstring(node),
            class_name=class_name,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            callees=sorted(_called_names(node) & (module_level - {node.name})),
        ))

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
    )


def render_context(ctx: ModuleContext, qualname: str, *, max_chars: int = 6000) -> str:
    """Prompt-ready context for one function, trimmed to `max_chars`."""
    fn = ctx.function(qualname)
    tree = ast.parse(ctx.source)
    helper_sigs = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in fn.callees:
            if isinstance(node, ast.ClassDef):
                helper_sigs.append(_class_header(node))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                first = f'\n    """{doc.splitlines()[0]}"""' if doc else ""
                helper_sigs.append(signature_of(node) + first + "\n    ...")
    parts = [
        f"# Module under test: {ctx.module_name}  (import with: {ctx.import_line(fn)})",
        "# Module imports:\n" + ("\n".join(ctx.imports) or "# (none)"),
    ]
    if helper_sigs:
        parts.append("# Helpers it calls (signatures only):\n" + "\n\n".join(helper_sigs))
    if fn.class_name:
        parts.append("# Enclosing class:\n" + ctx.class_headers[fn.class_name])
    parts.append("# Function under test:\n" + fn.source)
    text = "\n\n".join(parts)
    if len(text) > max_chars:  # keep the function itself; trim helper context first
        head = "\n\n".join(parts[:2])[: max(0, max_chars - len(parts[-1]) - 20)]
        text = head + "\n# ...(trimmed)\n\n" + parts[-1]
    return text


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


def _segment(source: str, start: int, end: int) -> str:
    return "\n".join(source.splitlines()[start - 1 : end]) + "\n"
