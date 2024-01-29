import pytest
from gotranx.expressions import build_expression
import sympy as sp


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("x = Conditional(Eq(time, 0), 1, 0)", (1, 0, 0)),
        ("x = Conditional(Gt(time, 0), 1, 0)", (0, 1, 0)),
        ("x = Conditional(Ge(time, 0), 1, 0)", (1, 1, 0)),
        ("x = Conditional(Lt(time, 0), 1, 0)", (0, 0, 1)),
        ("x = Conditional(Le(time, 0), 1, 0)", (1, 0, 1)),
        ("x = Conditional(Not(Eq(time, 0)), 1, 0)", (0, 1, 1)),
        ("x = Conditional(Not(Gt(time, 0)), 1, 0)", (1, 0, 1)),
        ("x = Conditional(Not(Ge(time, 0)), 1, 0)", (0, 0, 1)),
        ("x = Conditional(Not(Lt(time, 0)), 1, 0)", (1, 1, 0)),
        ("x = Conditional(Not(Le(time, 0)), 1, 0)", (0, 1, 0)),
    ],
)
def test_Conditional_expr(expr, expected, parser, trans):
    time = sp.Symbol("time")
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    sympy_expr = build_expression(result[0].value.tree, symbols={"time": time})
    assert sympy_expr.subs({"time": 0}) == expected[0]
    assert sympy_expr.subs({"time": 1}) == expected[1]
    assert sympy_expr.subs({"time": -1}) == expected[2]
