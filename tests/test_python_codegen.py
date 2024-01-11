import sys
from unittest import mock

import pytest
from gotranx.codegen import PythonCodeGenerator
from gotranx.codegen import RHSArgument
from gotranx.ode import make_ode


@pytest.fixture(scope="module")
def ode(trans, parser):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", x=1.0, y=2.0,z=3.05)

    expressions("My component")
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree), name="lorentz")


@pytest.fixture(scope="module")
def codegen(ode) -> PythonCodeGenerator:
    return PythonCodeGenerator(ode)


@pytest.fixture(scope="module")
def codegen_no_black(ode) -> PythonCodeGenerator:
    with mock.patch.dict(sys.modules, {"black": None}):
        return PythonCodeGenerator(ode)


def test_python_codegen_state_index_no_black(codegen: PythonCodeGenerator):
    assert codegen.state_index() == (
        "def state_index(name: str) -> float:"
        '\n    """Return the index of the state with the given name'
        "\n"
        "\n    Arguments"
        "\n    ---------"
        "\n    name : str"
        "\n        The name of the state"
        "\n"
        "\n    Returns"
        "\n    -------"
        "\n    float"
        "\n        The index of the state"
        "\n"
        "\n    Raises"
        "\n    ------"
        "\n    KeyError"
        "\n        If the name is not a valid state"
        '\n    """'
        "\n"
        '\n    data = {"x": 0, "y": 1, "z": 2}'
        "\n    return data[name]"
        "\n"
    )


def test_python_codegen_initial_state_values(codegen: PythonCodeGenerator):
    assert codegen.initial_state_values() == (
        "def init_state_values(**values):"
        '\n    """Initialize state values"""'
        "\n    # x=1.0, y=2.0, z=3.05"
        "\n"
        "\n    states = numpy.array([1.0, 2.0, 3.05])"
        "\n"
        "\n    for key, value in values.items():"
        "\n        states[state_index(key)] = value"
        "\n"
        "\n    return states"
        "\n"
    )


def test_python_codegen_parameter_index_no_black(codegen_no_black: PythonCodeGenerator):
    assert codegen_no_black.parameter_index() == (
        "\ndef parameter_index(name: str) -> float:"
        '\n    """Return the index of the parameter with the given name'
        "\n"
        "\n    Arguments"
        "\n    ---------"
        "\n    name : str"
        "\n        The name of the parameter"
        "\n"
        "\n    Returns"
        "\n    -------"
        "\n    float"
        "\n        The index of the parameter"
        "\n"
        "\n    Raises"
        "\n    ------"
        "\n    KeyError"
        "\n        If the name is not a valid parameter"
        '\n    """'
        "\n"
        "\n    data = {'a': 0, 'beta': 1, 'rho': 2, 'sigma': 3}"
        "\n    return data[name]"
        "\n"
    )


def test_python_codegen_parameter_index(codegen: PythonCodeGenerator):
    assert codegen.parameter_index() == (
        "def parameter_index(name: str) -> float:"
        '\n    """Return the index of the parameter with the given name'
        "\n"
        "\n    Arguments"
        "\n    ---------"
        "\n    name : str"
        "\n        The name of the parameter"
        "\n"
        "\n    Returns"
        "\n    -------"
        "\n    float"
        "\n        The index of the parameter"
        "\n"
        "\n    Raises"
        "\n    ------"
        "\n    KeyError"
        "\n        If the name is not a valid parameter"
        '\n    """'
        "\n"
        '\n    data = {"a": 0, "beta": 1, "rho": 2, "sigma": 3}'
        "\n    return data[name]"
        "\n"
    )


def test_python_codegen_initial_parameter_values(codegen: PythonCodeGenerator):
    assert codegen.initial_parameter_values() == (
        "def init_parameter_values(**values):"
        '\n    """Initialize parameter values"""'
        "\n    # a=0, beta=2.4, rho=21.0, sigma=12.0"
        "\n"
        "\n    parameters = numpy.array([0, 2.4, 21.0, 12.0])"
        "\n"
        "\n    for key, value in values.items():"
        "\n        parameters[parameter_index(key)] = value"
        "\n"
        "\n    return parameters"
        "\n"
    )


@pytest.mark.parametrize(
    "order, arguments",
    [
        (
            RHSArgument.stp,
            ("states, t, parameters"),
        ),
        (
            RHSArgument.spt,
            ("states, parameters, t"),
        ),
        (
            RHSArgument.tsp,
            ("t, states, parameters"),
        ),
    ],
)
def test_python_codegen_rhs(order: str, arguments: str, codegen: PythonCodeGenerator):
    assert codegen.rhs(order=order) == (
        f"def rhs({arguments}):"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    y = states[1]"
        "\n    z = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(3)"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[0] = dx_dt"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    values[1] = dy_dt"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    values[2] = dz_dt"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_codegen_forward_explicit_euler(codegen: PythonCodeGenerator):
    assert codegen.scheme("forward_explicit_euler") == (
        "def forward_explicit_euler(states, t, dt, parameters):"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    y = states[1]"
        "\n    z = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(3)"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[0] = dt * dx_dt + x"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    values[1] = dt * dy_dt + y"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    values[2] = dt * dz_dt + z"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_codegen_forward_generalized_rush_larsen(codegen: PythonCodeGenerator):
    assert codegen.scheme("forward_generalized_rush_larsen") == (
        "def forward_generalized_rush_larsen(states, t, dt, parameters):"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    y = states[1]"
        "\n    z = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(3)"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    dx_dt_linearized = -sigma"
        "\n    values[0] = ("
        "\n        x"
        "\n        + dx_dt"
        "\n        * (numpy.exp(dt * dx_dt_linearized) - 1)"
        "\n        / dx_dt_linearized"
        "\n        * (numpy.abs(dx_dt_linearized) > 1e-08)"
        "\n        + dt * dx_dt * (~(numpy.abs(dx_dt_linearized) > 1e-08))"
        "\n    )"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    dy_dt_linearized = -1"
        "\n    values[1] = ("
        "\n        y"
        "\n        + dy_dt"
        "\n        * (numpy.exp(dt * dy_dt_linearized) - 1)"
        "\n        / dy_dt_linearized"
        "\n        * (numpy.abs(dy_dt_linearized) > 1e-08)"
        "\n        + dt * dy_dt * (~(numpy.abs(dy_dt_linearized) > 1e-08))"
        "\n    )"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    dz_dt_linearized = -beta"
        "\n    values[2] = ("
        "\n        z"
        "\n        + dz_dt"
        "\n        * (numpy.exp(dt * dz_dt_linearized) - 1)"
        "\n        / dz_dt_linearized"
        "\n        * (numpy.abs(dz_dt_linearized) > 1e-08)"
        "\n        + dt * dz_dt * (~(numpy.abs(dz_dt_linearized) > 1e-08))"
        "\n    )"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_conditional_expression(parser, trans):
    expr = """
    states(v=0)
    ah = Conditional(Ge(v, -40), 0, 0.057*exp(-(v + 80)/6.8))
    dv_dt = 0
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = make_ode(*result, name="conditional")
    codegen = PythonCodeGenerator(ode)
    assert codegen.rhs() == (
        "def rhs(t, states, parameters):"
        "\n    # Assign states"
        "\n    v = states[0]"
        "\n"
        "\n    # Assign parameters"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(1)"
        "\n    dv_dt = 0"
        "\n    values[0] = dv_dt"
        "\n    ah = (0) * (v >= -40) + (0.057 * numpy.exp((-(v + 80)) / 6.8)) * numpy.logical_not("
        "\n        (v >= -40)"
        "\n    )"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_exponential_with_power(parser, trans):
    expr = """
    states(v=0)
    tm = 0.06487*exp(-((v - 4.823)/51.12)**2)
    dv_dt = 0
    """
    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = make_ode(*result, name="power")
    codegen = PythonCodeGenerator(ode)

    assert codegen.rhs() == (
        "def rhs(t, states, parameters):"
        "\n    # Assign states"
        "\n    v = states[0]"
        "\n"
        "\n    # Assign parameters"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(1)"
        "\n    dv_dt = 0"
        "\n    values[0] = dv_dt"
        "\n    tm = 0.06487 * numpy.exp(-(((v - 4.823) / 51.12) ** 2))"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_remove_unused_rhs(parser, trans):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4,
    unused_parameter=1.0
    )
    states("My component", x=1.0, y=2.0,z=3.05, unused_state=1.0)

    expressions("My component")
    dunused_state_dt = 0
    unused_expression = 0
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))
    codegen_orig = PythonCodeGenerator(ode)
    assert codegen_orig.rhs() == (
        "def rhs(t, states, parameters):"
        "\n    # Assign states"
        "\n    unused_state = states[0]"
        "\n    x = states[1]"
        "\n    y = states[2]"
        "\n    z = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n    unused_parameter = parameters[4]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(4)"
        "\n    unused_expression = 0"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dunused_state_dt"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[1] = dx_dt"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    values[2] = dy_dt"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    values[3] = dz_dt"
        "\n"
        "\n    return values"
        "\n"
    )
    codegen_remove = PythonCodeGenerator(ode, remove_unused=True)
    assert codegen_remove.rhs() == (
        "def rhs(t, states, parameters):"
        "\n    # Assign states"
        "\n    x = states[1]"
        "\n    y = states[2]"
        "\n    z = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(4)"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dunused_state_dt"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[1] = dx_dt"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    values[2] = dy_dt"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    values[3] = dz_dt"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_remove_unused_forward_explicit_euler(parser, trans):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4,
    unused_parameter=1.0
    )
    states("My component", x=1.0, y=2.0,z=3.05, unused_state=1.0)

    expressions("My component")
    dunused_state_dt = 0
    unused_expression = 0
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))
    codegen_orig = PythonCodeGenerator(ode)
    assert codegen_orig.scheme("forward_explicit_euler") == (
        "def forward_explicit_euler(states, t, dt, parameters):"
        "\n    # Assign states"
        "\n    unused_state = states[0]"
        "\n    x = states[1]"
        "\n    y = states[2]"
        "\n    z = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n    unused_parameter = parameters[4]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(4)"
        "\n    unused_expression = 0"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dt * dunused_state_dt + unused_state"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[1] = dt * dx_dt + x"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    values[2] = dt * dy_dt + y"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    values[3] = dt * dz_dt + z"
        "\n"
        "\n    return values"
        "\n"
    )
    codegen_remove = PythonCodeGenerator(ode, remove_unused=True)
    assert codegen_remove.scheme("forward_explicit_euler") == (
        "def forward_explicit_euler(states, t, dt, parameters):"
        "\n    # Assign states"
        "\n    unused_state = states[0]"
        "\n    x = states[1]"
        "\n    y = states[2]"
        "\n    z = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(4)"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dt * dunused_state_dt + unused_state"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[1] = dt * dx_dt + x"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    values[2] = dt * dy_dt + y"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    values[3] = dt * dz_dt + z"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_remove_unused_forward_generalized_rush_larsen(parser, trans):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4,
    unused_parameter=1.0
    )
    states("My component", x=1.0, y=2.0,z=3.05, unused_state=1.0)

    expressions("My component")
    dunused_state_dt = 0
    unused_expression = 0
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))
    codegen_orig = PythonCodeGenerator(ode)
    assert codegen_orig.scheme("forward_generalized_rush_larsen") == (
        "def forward_generalized_rush_larsen(states, t, dt, parameters):"
        "\n    # Assign states"
        "\n    unused_state = states[0]"
        "\n    x = states[1]"
        "\n    y = states[2]"
        "\n    z = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n    unused_parameter = parameters[4]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(4)"
        "\n    unused_expression = 0"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dt * dunused_state_dt + unused_state"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    dx_dt_linearized = -sigma"
        "\n    values[1] = ("
        "\n        x"
        "\n        + dx_dt"
        "\n        * (numpy.exp(dt * dx_dt_linearized) - 1)"
        "\n        / dx_dt_linearized"
        "\n        * (numpy.abs(dx_dt_linearized) > 1e-08)"
        "\n        + dt * dx_dt * (~(numpy.abs(dx_dt_linearized) > 1e-08))"
        "\n    )"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    dy_dt_linearized = -1"
        "\n    values[2] = ("
        "\n        y"
        "\n        + dy_dt"
        "\n        * (numpy.exp(dt * dy_dt_linearized) - 1)"
        "\n        / dy_dt_linearized"
        "\n        * (numpy.abs(dy_dt_linearized) > 1e-08)"
        "\n        + dt * dy_dt * (~(numpy.abs(dy_dt_linearized) > 1e-08))"
        "\n    )"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    dz_dt_linearized = -beta"
        "\n    values[3] = ("
        "\n        z"
        "\n        + dz_dt"
        "\n        * (numpy.exp(dt * dz_dt_linearized) - 1)"
        "\n        / dz_dt_linearized"
        "\n        * (numpy.abs(dz_dt_linearized) > 1e-08)"
        "\n        + dt * dz_dt * (~(numpy.abs(dz_dt_linearized) > 1e-08))"
        "\n    )"
        "\n"
        "\n    return values"
        "\n"
    )
    codegen_remove = PythonCodeGenerator(ode, remove_unused=True)
    assert codegen_remove.scheme("forward_generalized_rush_larsen") == (
        "def forward_generalized_rush_larsen(states, t, dt, parameters):"
        "\n    # Assign states"
        "\n    unused_state = states[0]"
        "\n    x = states[1]"
        "\n    y = states[2]"
        "\n    z = states[3]"
        "\n"
        "\n    # Assign parameters"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    values = numpy.zeros(4)"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dt * dunused_state_dt + unused_state"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    dx_dt_linearized = -sigma"
        "\n    values[1] = ("
        "\n        x"
        "\n        + dx_dt"
        "\n        * (numpy.exp(dt * dx_dt_linearized) - 1)"
        "\n        / dx_dt_linearized"
        "\n        * (numpy.abs(dx_dt_linearized) > 1e-08)"
        "\n        + dt * dx_dt * (~(numpy.abs(dx_dt_linearized) > 1e-08))"
        "\n    )"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    dy_dt_linearized = -1"
        "\n    values[2] = ("
        "\n        y"
        "\n        + dy_dt"
        "\n        * (numpy.exp(dt * dy_dt_linearized) - 1)"
        "\n        / dy_dt_linearized"
        "\n        * (numpy.abs(dy_dt_linearized) > 1e-08)"
        "\n        + dt * dy_dt * (~(numpy.abs(dy_dt_linearized) > 1e-08))"
        "\n    )"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    dz_dt_linearized = -beta"
        "\n    values[3] = ("
        "\n        z"
        "\n        + dz_dt"
        "\n        * (numpy.exp(dt * dz_dt_linearized) - 1)"
        "\n        / dz_dt_linearized"
        "\n        * (numpy.abs(dz_dt_linearized) > 1e-08)"
        "\n        + dt * dz_dt * (~(numpy.abs(dz_dt_linearized) > 1e-08))"
        "\n    )"
        "\n"
        "\n    return values"
        "\n"
    )
