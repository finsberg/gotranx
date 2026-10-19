import pytest
import sympy
from sympy.codegen.ast import Assignment

from gotranx.codegen.cpp import GotranCppCodePrinter

x, y = sympy.symbols("x y")


def test_cpp_printer_uses_std_math_and_fmax():
    p = GotranCppCodePrinter()
    assert p.doprint(sympy.exp(x)) == "std::exp(x)"
    # std::max(0, V) does not compile (int vs double), so Max stays fmax
    assert p.doprint(sympy.Max(0, x)) == "std::fmax(0, x)"
    assert p.doprint(sympy.Min(y, 1)) == "std::fmin(1, y)"


def test_cpp_printer_keeps_the_piecewise_assignment_ternary():
    expr = sympy.Piecewise((x, x > 0), (2 * x, True))
    assert GotranCppCodePrinter().doprint(Assignment(y, expr)) == "y = (x > 0) ? x : 2*x;"


def test_cpp_printer_double_literals():
    p = GotranCppCodePrinter()
    assert p.doprint(sympy.Float(2.5) * x) == "2.5*x"
    assert p.doprint(sympy.Rational(1, 3)) == "1.0/3.0"


@pytest.mark.parametrize(
    "expr, expected",
    [
        (sympy.Float(2.5) * x, "2.5f*x"),
        (x**2, "std::pow(x, 2.0f)"),
        (sympy.Rational(1, 3), "1.0f/3.0f"),
        (sympy.Max(0, x), "std::fmax(0.0f, x)"),
        (sympy.IndexedBase("states", shape=(3,))[0], "states[0]"),
    ],
)
def test_cpp_printer_float_literals(expr, expected):
    assert GotranCppCodePrinter(float_type="float").doprint(expr) == expected


def test_cpp_printer_renames_cxx_keywords():
    assert GotranCppCodePrinter().doprint(sympy.Symbol("new") * 2) == "2*new_"
