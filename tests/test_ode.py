import pytest
from gotranx import exceptions
from gotranx import ode


def test_ODE_with_incomplete_component_raises_ComponentNotCompleteError(parser, trans):
    expr = "states(x=1, y=2)\n dy_dt=0"
    tree = parser.parse(expr)
    result = trans.transform(tree)
    with pytest.raises(exceptions.ComponentNotCompleteError) as e:
        ode.ODE(result.components)
    assert str(e.value) == "Component '' is not complete. Missing state derivatives for ['x']"


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
        ode.ODE(trans.transform(tree).components)

    assert "Found multiple definitions for {'y'}" in str(e.value)


def test_make_ode_with_duplicates_raises_DuplicateSymbolError(parser, trans):
    expr = "states(y=2)\n dy_dt=0 \n y=42"
    tree = parser.parse(expr)
    with pytest.raises(exceptions.DuplicateSymbolError) as e:
        ode.make_ode(trans.transform(tree).components)

    assert "Found multiple definitions for {'y'}" in str(e.value)


def test_make_ode(parser, trans):
    expr = """
    # This is an ODE.
    # Here is another line.
    states("First component", "X-gate", x = 1, xr=3.14)
    states("First component", "Y-gate", y = 1)
    states("Second component", z=1)
    parameters("First component", a=1, b=2)
    parameters("Second component", c=3)

    expressions("First component")
    d = a + b * 2 - 3 / c

    expressions("First component", "X-gate")
    dx_dt=a+1
    dxr_dt = (x / b) - d * xr

    expressions("First component", "Y-gate")
    dy_dt = 2 * d - 1

    expressions("Second component")
    dz_dt = 1 + x - y
    """
    tree = parser.parse(expr)

    result = ode.make_ode(*trans.transform(tree), name="TestODE")
    assert result.text == "This is an ODE. Here is another line."
    assert repr(result) == "ODE(TestODE, num_states=4, num_parameters=3)"

    assert result.num_components == 4

    states = result.states
    assert len(states) == result.num_states == 4
    state_names = set()
    state_symbols = set()
    for state in states:
        # assert isinstance(state, atoms.TimeDependentState)
        state_names.add(state.name)
        state_symbols.add(str(state.symbol))

    assert state_names == {"x", "xr", "y", "z"}
    # assert state_symbols == {"x(t)", "xr(t)", "y(t)", "z(t)"}

    parameters = result.parameters
    assert len(parameters) == result.num_parameters == 3
    parameter_names = set()
    for parameter in parameters:
        parameter_names.add(parameter.name)

    assert parameter_names == {"a", "b", "c"}

    state_derivatives = result.state_derivatives
    assert len(state_derivatives) == 4
    state_derivative_names = set()
    state_derivative_symbols = set()
    for state_derivative in state_derivatives:
        # assert isinstance(state_derivative.state, atoms.TimeDependentState)
        state_derivative_names.add(state_derivative.name)
        state_derivative_symbols.add(str(state_derivative.symbol))

    assert state_derivative_names == {"dx_dt", "dxr_dt", "dy_dt", "dz_dt"}
    # assert state_derivative_symbols == {
    #     "Derivative(x(t), t)",
    #     "Derivative(xr(t), t)",
    #     "Derivative(y(t), t)",
    #     "Derivative(z(t), t)",
    # }

    intermediates = result.intermediates
    assert len(intermediates) == 1


@pytest.mark.parametrize(
    "assignments_only, expected",
    [
        (True, ("y", "a", "z", "x", "dV_dt")),
        (False, ("V", "c", "y", "a", "z", "x", "dV_dt")),
    ],
)
def test_sort_assignment(assignments_only, expected, parser, trans):
    expr = """
    states(V=0)
    parameters(c=1)
    dV_dt = x * z
    x = y + z + V
    z = a - 2 * c
    a = y + 3
    y = 2 * V
    """
    tree = parser.parse(expr)
    result = ode.make_ode(*trans.transform(tree), name="TestODE")

    sorted_assignments = result.sorted_assignments(
        assignments_only=assignments_only,
    )
    assert tuple([x.name for x in sorted_assignments]) == expected


def test_ode_dependents(parser, trans):
    expr = """
    states(V=0)
    dV_dt = x * z
    x = y + z + V
    z = 2
    a = y + 3
    y = 2 * V
    """
    tree = parser.parse(expr)
    result = ode.make_ode(*trans.transform(tree), name="TestODE")
    deps = result.dependents()
    assert deps["V"] == {"x", "y"}
    assert deps["x"] == {"dV_dt"}
    assert deps["z"] == {"dV_dt", "x"}
    assert deps["y"] == {"a", "x"}
    assert "dV_dt" not in deps
    assert "a" not in deps
