import pytest
from gotranx.expressions import build_expression
from gotranx import codegen
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


@pytest.mark.parametrize(
    "Printer, expected",
    [
        (codegen.BaseGotranODECodePrinter, "Conditional(Lt(t, 0), a, b)"),
        (codegen.GotranPythonCodePrinter, "numpy.where((t < 0), a, b)"),
        (codegen.GotranCCodePrinter, "((t < 0) ? (\n   a\n)\n: (\n   b\n))"),
    ],
)
def test_single_Conditional_from_sympy(Printer, expected):
    t = sp.Symbol("t")
    a = sp.Symbol("a")
    b = sp.Symbol("b")

    expr = sp.Piecewise((a, sp.Lt(t, 0)), (b, True))
    result = Printer().doprint(expr)
    assert result == expected


@pytest.mark.parametrize(
    "Printer, expected",
    [
        (
            codegen.BaseGotranODECodePrinter,
            "Conditional(Lt(t, 0), a, Conditional(Gt(t, 0), b, c))",
        ),
        (
            codegen.GotranPythonCodePrinter,
            "numpy.where((t < 0), a, numpy.where((t > 0), b, c))",
        ),
        (
            codegen.GotranCCodePrinter,
            "((t < 0) ? (\n   a\n)\n: ((t > 0) ? (\n   b\n)\n: (\n   c\n)))",
        ),
    ],
)
def test_nested_Conditional_from_sympy(Printer, expected):
    t = sp.Symbol("t")
    a = sp.Symbol("a")
    b = sp.Symbol("b")
    c = sp.Symbol("c")

    expr = sp.Piecewise((a, t < 0), (b, t > 0), (c, True))
    result = Printer().doprint(expr)

    assert result == expected


def test_And_Or_from_sympy():
    t = sp.Symbol("t")
    expr = sp.Or(sp.Lt(t, 0), sp.And(sp.Gt(t, 0), sp.Lt(t, 1)))
    result = codegen.BaseGotranODECodePrinter().doprint(expr)

    assert result == "Or(Lt(t, 0), And(Gt(t, 0), Lt(t, 1)))"
