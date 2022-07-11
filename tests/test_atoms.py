import pytest


@pytest.mark.parametrize(
    "expr, deps",
    [
        ("x = (1 * y) + rho - (z / sigma)", {"y", "rho", "z", "sigma"}),
        ("x = (1 * y) + rho - (y / 2)", {"y", "rho"}),
        ("x = 1", set()),
    ],
)
def test_expression_dependencies(expr, deps, parser, trans):

    tree = parser.parse(expr)
    result = trans.transform(tree)
    assert result.rhs.dependencies == deps
