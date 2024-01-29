import gotranx
import pytest
import sympy as sp
from gotranx.expressions import build_expression


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

    assert sympy_expr.subs(symbol_values).evalf() == pytest.approx(expected)


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
    t = sp.Symbol("t")

    old_ode = gotranx.ode.ODE(result.components)

    symbols = old_ode.symbols
    ode = gotranx.ode.ODE(
        gotranx.ode.resolve_expressions(result.components, symbols=symbols),
        t=t,
    )

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

    assert ode["x"].expr.evalf() == ((a + 1) / b).evalf()
    assert ode["y"].expr.evalf() == (b * x).evalf()
    assert ode["du_dt"].expr.evalf() == (x + y - u).evalf()


def test_add_temporal_state(parser, trans):
    expr = """
    states(u=42)
    parameters(a=1, b=2)
    x = (1 + a) / b
    y = b  * x
    du_dt = y + x - u
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    t = sp.Symbol("t")
    components = gotranx.ode.add_temporal_state(components=result.components, t=t)
    ode = gotranx.ode.ODE(components=components, t=t)
    assert str(ode["u"].symbol) == "u(t)"
    assert str(ode["du_dt"].symbol) == "Derivative(u(t), t)"


def test_state_parameters_with_expression_values(parser, trans):
    expr = """
    states(u=2*4/3)
    parameters(a=3*5**2)
    x = (1 + a) / b
    y = b  * x
    du_dt = y + x - u
    """
    tree = parser.parse(expr)
    result = trans.transform(tree).components[0]

    assert result.find_state("u").value.args == (
        sp.Mul(2, 4, evaluate=False),
        sp.Pow(3, sp.Integer(-1), evaluate=False),
    )
    assert result.find_parameter("a").value == sp.Mul(
        3, sp.Pow(5, 2, evaluate=False), evaluate=False
    )


def test_expression_missing_symbol(parser, trans):
    expr = "x = a + 1"
    tree = parser.parse(expr)
    result = trans.transform(tree)
    with pytest.raises(gotranx.expressions.MissingSymbolError) as e:
        build_expression(result[0].value.tree, symbols={})
    assert str(e.value) == "Symbol 'a' not found in line 1"
