import pytest
import sympy as sp
from gotran_parser import ODE
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


def test_build_two_expressions(parser, trans):
    expr = """
    states(u=42)
    parameters(a=1, b=2)
    x = (1 + a) / b
    y = b  * x
    du_dt = y + x - u
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = ODE(result).resolve_expressions()
    assert len(ode.components) == 1
    assert len(ode.components[0].parameters) == 2
    assert len(ode.components[0].states) == 1
    assert len(ode.components[0].assignments) == 3
    assert len(ode.components[0].intermediates) == 2
    assert len(ode.components[0].state_derivatives) == 1

    u = ode["u"].symbol
    a = ode["a"].symbol
    b = ode["b"].symbol
    x = ode["x"].symbol
    y = ode["y"].symbol

    assert ode["x"].expr == (a + 1.0) / b
    assert ode["y"].expr == b * x
    assert ode["du_dt"].expr == x + y - u
