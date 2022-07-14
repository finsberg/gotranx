import pytest
from gotran_parser import atoms


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
    assert result[0].rhs.dependencies == deps


def test_parameter_arguments(parser, trans):

    expr = """parameters("My component",
        y = 1.0,
        x = ScalarParam(42.0),
        w = ScalarParam(10.2, unit="mM"),
        v = ScalarParam(3.14, unit="mV", description="Description of v"),
        z = ScalarParam(42.0, description="Description of z")
    )
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)

    assert result[0] == atoms.Parameter(name="y", value=1.0, component="My component")
    assert result[1] == atoms.Parameter(name="x", value=42.0, component="My component")
    assert result[2] == atoms.Parameter(
        name="w",
        value=10.2,
        component="My component",
        unit_str="mM",
    )
    assert result[3] == atoms.Parameter(
        name="v",
        value=3.14,
        description="Description of v",
        component="My component",
        unit_str="mV",
    )
    assert result[4] == atoms.Parameter(
        name="z",
        value=42.0,
        description="Description of z",
        component="My component",
    )
