import pytest
import math
from gotranx.expressions import build_expression


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("x = Abs(-1)", 1),
        ("x = Abs(1)", 1),
        ("x = Abs(0)", 0),
        ("x = Abs(-0)", 0),
        ("x = abs(-1)", 1),
        ("x = abs(1)", 1),
        ("x = abs(0)", 0),
        ("x = abs(-0)", 0),
        ("x = cos(0)", 1),
        ("x = cos(pi/2)", 0),
        ("x = cos(pi)", -1),
        ("x = sin(0)", 0),
        ("x = sin(pi/2)", 1),
        ("x = sin(pi)", 0),
        ("x = tan(0)", 0),
        ("x = tan(pi)", 0),
        ("x = tan(pi/4)", 1),
        ("x = tan(pi)", 0),
        ("x = ln(1)", 0),
        ("x = ln(10)", math.log(10)),
        ("x = sqrt(1)", 1),
        ("x = sqrt(4)", 2),
        ("x = sqrt(9)", 3),
        ("x = exp(0)", 1),
        ("x = exp(1)", math.exp(1)),
        ("x = log(1)", 0),
        ("x = log(exp(1))", 1),
        ("x = ln(1)", 0),
        ("x = ln(10)", math.log(10)),
        ("x = asin(0)", 0),
        ("x = asin(1)", math.pi / 2),
        ("x = acos(0)", math.pi / 2),
        ("x = acos(1)", 0),
        ("x = atan(0)", 0),
        ("x = atan(1)", math.pi / 4),
        ("x = floor(1.1)", 1),
        ("x = floor(1.9)", 1),
        ("x = Mod(10, 3)", 1),
        ("x = Mod(10, 2)", 0),
        ("x = Mod(10, 5)", 0),
        ("x = Mod(10, 4)", 2),
        ("x = Mod(10, 7)", 3),
    ],
)
def test_operations(expr, expected, parser, trans):
    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert len(result) == 1
    sympy_expr = build_expression(result[0].value.tree)
    assert math.isclose(float(sympy_expr), expected)
