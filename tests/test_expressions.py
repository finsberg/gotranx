import pytest
import sympy as sp
from gotran_parser.expressions import build_expression


@pytest.mark.parametrize(
    "expr, symbol_values, expected",
    [
        ("x = y + 1 ", {"y": 2}, 3),
        ("x = y - 1 ", {"y": 2}, 1),
        ("x = y * 2 ", {"y": 2}, 4),
        ("x = y / 2 ", {"y": 4}, 2),
        ("x = y * (a + b - c) / 2 ", {"y": 2, "a": 3, "b": 6, "c": 1}, 8),
    ],
)
def test_build_expression(expr, symbol_values, expected, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)

    symbols = {name: sp.Symbol(name) for name in symbol_values}
    value = result[0].value.tree
    sympy_expr = build_expression(value, symbols)
    assert sympy_expr.subs(symbol_values) == pytest.approx(expected)
