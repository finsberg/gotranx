import pytest
from gotran_parser import exceptions
from gotran_parser import ode


def test_ODE_with_incomplete_component_raises_ComponentNotCompleteError(parser, trans):
    expr = "states(x=1, y=2)\n dy_dt=0"
    tree = parser.parse(expr)
    result = trans.transform(tree)
    with pytest.raises(exceptions.ComponentNotCompleteError) as e:
        ode.ODE(result)
    assert (
        str(e.value)
        == "Component None is not complete. Missing state derivatives for ['x']"
    )


@pytest.mark.parametrize(
    "input, output",
    [
        ((), set()),
        (("x", "y", "z"), set()),
        (("x", "y", "y"), {"y"}),
        (("x"), set()),
        (("y", "y", "y"), {"y"}),
        (("x", "y", "y", "x"), {"x", "y"}),
    ],
)
def test_find_duplicates(input, output):
    assert ode.find_duplicates(input) == output


def test_ODE_with_duplicates_raises_DuplicateSymbolError(parser, trans):
    expr = "states(y=2)\n dy_dt=0 \n y=42"
    tree = parser.parse(expr)
    with pytest.raises(exceptions.DuplicateSymbolError) as e:
        ode.ODE(trans.transform(tree))

    assert "Found multiple definitions for {'y'}" in str(e.value)


def test_ODE_resolve_expressions(parser, trans):
    expr = """
    states("First component", "X-gate", x = 1, xr=3.14)
    states("First component", "Y-gate", y = 1)
    states("Second component", z=1)
    parameters("First component", a=1, b=2)
    parameters("Second component", c=3)

    expressions("First component")
    dx_dt=a+1
    d = a + b * 2 - 3 / c
    dy_dt = 2 * d - 1
    dxr_dt = (x / b) - d * xr

    expressions("Second component")
    dz_dt = 1 + x - y
    """
    tree = parser.parse(expr)
    result = ode.ODE(trans.transform(tree))
    result.resolve_expressions()
