from pathlib import Path

import pytest

from autocover.tools.context import build_module_context, module_name_for, render_context

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = ROOT / "examples" / "ticket_price"


def test_module_name_handles_src_layout_and_packages(tmp_path):
    (tmp_path / "src" / "pkg" / "sub").mkdir(parents=True)
    assert module_name_for(tmp_path, tmp_path / "src/pkg/sub/mod.py") == "pkg.sub.mod"
    assert module_name_for(tmp_path, tmp_path / "src/pkg/__init__.py") == "pkg"
    assert module_name_for(tmp_path, tmp_path / "flat.py") == "flat"


def test_functions_and_callees_of_example():
    ctx = build_module_context(EXAMPLE_DIR, "ticket_price.py")
    assert ctx.module_name == "ticket_price"
    assert [f.qualname for f in ctx.functions] == ["ticket_price", "group_total"]
    group = ctx.function("group_total")
    assert group.callees == ["ticket_price"]
    assert group.signature == "def group_total(ages: list[int], discount: float=0.0) -> float:"


def test_render_context_includes_helper_signature_not_body():
    ctx = build_module_context(EXAMPLE_DIR, "ticket_price.py")
    text = render_context(ctx, "group_total")
    assert "from ticket_price import group_total" in text
    assert "def ticket_price(age: int) -> int:" in text
    assert "raise ValueError" not in text  # helper body omitted
    assert "total = sum(ticket_price(a) for a in ages)" in text


SOURCE_WITH_CLASS = '''
import math


class Cart:
    def __init__(self, items: list[float]):
        self.items = items

    def total(self) -> float:
        return math.fsum(self.items)

    def _private(self):
        return 1


def _hidden():
    return 0
'''


def test_methods_get_class_header_and_private_are_skipped(tmp_path):
    (tmp_path / "cart.py").write_text(SOURCE_WITH_CLASS)
    ctx = build_module_context(tmp_path, "cart.py")
    assert [f.qualname for f in ctx.functions] == ["Cart.total"]
    text = render_context(ctx, "Cart.total")
    assert "from cart import Cart" in text
    assert "def __init__(self, items: list[float]):" in text
    with_private = build_module_context(tmp_path, "cart.py", include_private=True)
    assert "_hidden" in [f.qualname for f in with_private.functions]


def test_unknown_function_raises():
    ctx = build_module_context(EXAMPLE_DIR, "ticket_price.py")
    with pytest.raises(KeyError):
        ctx.function("nope")
