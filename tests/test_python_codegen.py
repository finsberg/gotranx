import sys
from unittest import mock

import pytest
from gotranx.schemes import get_scheme
from gotranx.codegen import PythonCodeGenerator, JaxCodeGenerator
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
    rhoz = rho - z
    dy_dt = x*rhoz - y # millivolt
    dx_dt = sigma*(-x + y)
    betaz = beta*z
    dz_dt = -betaz + x*y
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree), name="lorentz")


@pytest.fixture(scope="module")
def codegen(ode) -> PythonCodeGenerator:
    return PythonCodeGenerator(ode)


@pytest.fixture(scope="module")
def codegen_jax(ode) -> JaxCodeGenerator:
    return JaxCodeGenerator(ode)


@pytest.fixture(scope="module")
def codegen_no_formatter(ode) -> PythonCodeGenerator:
    with mock.patch.dict(sys.modules, {"black": None, "ruff": None}):
        return PythonCodeGenerator(ode)


def test_python_codegen_state_index_no_formatter(codegen: PythonCodeGenerator):
    assert codegen.state_index() == (
        'state = {"x": 0, "z": 1, "y": 2}'
        "\n"
        "\n"
        "\ndef state_index(name: str) -> int:"
        '\n    """Return the index of the state with the given name'
        "\n"
        "\n    Arguments"
        "\n    ---------"
        "\n    name : str"
        "\n        The name of the state"
        "\n"
        "\n    Returns"
        "\n    -------"
        "\n    int"
        "\n        The index of the state"
        "\n"
        "\n    Raises"
        "\n    ------"
        "\n    KeyError"
        "\n        If the name is not a valid state"
        '\n    """'
        "\n"
        "\n    return state[name]"
        "\n"
    )


def test_python_codegen_initial_state_values(codegen: PythonCodeGenerator):
    assert codegen.initial_state_values() == (
        "def init_state_values(**values):"
        '\n    """Initialize state values"""'
        "\n    # x=1.0, z=3.05, y=2.0"
        "\n"
        "\n    states = numpy.array([1.0, 3.05, 2.0], dtype=numpy.float64)"
        "\n"
        "\n    for key, value in values.items():"
        "\n        states[state_index(key)] = value"
        "\n"
        "\n    return states"
        "\n"
    )


def test_python_codegen_parameter_index_no_formatter(codegen_no_formatter: PythonCodeGenerator):
    assert codegen_no_formatter.parameter_index() == (
        "\nparameter = {'a': 0, 'beta': 1, 'rho': 2, 'sigma': 3}"
        "\n"
        "\n"
        "\ndef parameter_index(name: str) -> int:"
        '\n    """Return the index of the parameter with the given name'
        "\n"
        "\n    Arguments"
        "\n    ---------"
        "\n    name : str"
        "\n        The name of the parameter"
        "\n"
        "\n    Returns"
        "\n    -------"
        "\n    int"
        "\n        The index of the parameter"
        "\n"
        "\n    Raises"
        "\n    ------"
        "\n    KeyError"
        "\n        If the name is not a valid parameter"
        '\n    """'
        "\n"
        "\n    return parameter[name]"
        "\n"
    )


def test_python_codegen_parameter_index(codegen: PythonCodeGenerator):
    assert codegen.parameter_index() == (
        'parameter = {"a": 0, "beta": 1, "rho": 2, "sigma": 3}'
        "\n"
        "\n"
        "\ndef parameter_index(name: str) -> int:"
        '\n    """Return the index of the parameter with the given name'
        "\n"
        "\n    Arguments"
        "\n    ---------"
        "\n    name : str"
        "\n        The name of the parameter"
        "\n"
        "\n    Returns"
        "\n    -------"
        "\n    int"
        "\n        The index of the parameter"
        "\n"
        "\n    Raises"
        "\n    ------"
        "\n    KeyError"
        "\n        If the name is not a valid parameter"
        '\n    """'
        "\n"
        "\n    return parameter[name]"
        "\n"
    )


def test_python_codegen_initial_parameter_values(codegen: PythonCodeGenerator):
    assert codegen.initial_parameter_values() == (
        "def init_parameter_values(**values):"
        '\n    """Initialize parameter values"""'
        "\n    # a=0, beta=2.4, rho=21.0, sigma=12.0"
        "\n"
        "\n    parameters = numpy.array([0, 2.4, 21.0, 12.0], dtype=numpy.float64)"
        "\n"
        "\n    for key, value in values.items():"
        "\n        parameters[parameter_index(key)] = value"
        "\n"
        "\n    return parameters"
        "\n"
    )


def test_python_codegen_initial_parameter_values_no_parameters(trans, parser):
    expr = """
    states(x=1, y=0)

    dx_dt = y
    dy_dt = -x
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree), name="lorentz")
    codegen = PythonCodeGenerator(ode)
    assert codegen.initial_parameter_values() == (
        "def init_parameter_values(**values):"
        '\n    """Initialize parameter values"""'
        "\n"
        "\n    parameters = numpy.array([], dtype=numpy.float64)"
        "\n"
        "\n    for key, value in values.items():"
        "\n        parameters[parameter_index(key)] = value"
        "\n"
        "\n    return parameters"
        "\n"
    )


def test_python_codegen_missing_index_is_empty(codegen: PythonCodeGenerator):
    assert codegen.missing_index() == ""


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
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    z = states[1]"
        "\n    y = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    betaz = beta * z"
        "\n    rhoz = rho - z"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[0] = dx_dt"
        "\n    dz_dt = -betaz + x * y"
        "\n    values[1] = dz_dt"
        "\n    dy_dt = rhoz * x - y"
        "\n    values[2] = dy_dt"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_codegen_forward_explicit_euler(codegen: PythonCodeGenerator):
    assert codegen.scheme(get_scheme("forward_explicit_euler")) == (
        "def forward_explicit_euler(states, t, dt, parameters):"
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    z = states[1]"
        "\n    y = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    betaz = beta * z"
        "\n    rhoz = rho - z"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[0] = dt * dx_dt + x"
        "\n    dz_dt = -betaz + x * y"
        "\n    values[1] = dt * dz_dt + z"
        "\n    dy_dt = rhoz * x - y"
        "\n    values[2] = dt * dy_dt + y"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_codegen_forward_generalized_rush_larsen(codegen: PythonCodeGenerator):
    assert codegen.scheme(get_scheme("forward_generalized_rush_larsen")) == (
        "def forward_generalized_rush_larsen(states, t, dt, parameters):"
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    z = states[1]"
        "\n    y = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    betaz = beta * z"
        "\n    rhoz = rho - z"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    dx_dt_linearized = -sigma"
        "\n    values[0] = x + numpy.where("
        "\n        numpy.logical_or((dx_dt_linearized > 1e-08), (dx_dt_linearized < -1e-08)),"
        "\n        dx_dt * (numpy.exp(dt * dx_dt_linearized) - 1) / dx_dt_linearized,"
        "\n        dt * dx_dt,"
        "\n    )"
        "\n    dz_dt = -betaz + x * y"
        "\n    values[1] = dt * dz_dt + z"
        "\n    dy_dt = rhoz * x - y"
        "\n    dy_dt_linearized = -1"
        "\n    values[2] = y + numpy.where("
        "\n        numpy.logical_or((dy_dt_linearized > 1e-08), (dy_dt_linearized < -1e-08)),"
        "\n        dy_dt * (numpy.exp(dt * dy_dt_linearized) - 1) / dy_dt_linearized,"
        "\n        dt * dy_dt,"
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
        "\n"
        "\n    # Assign states"
        "\n    v = states[0]"
        "\n"
        "\n    # Assign parameters"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    dv_dt = 0"
        "\n    values[0] = dv_dt"
        "\n    ah = numpy.where((v >= -40), 0, 0.057 * numpy.exp((-(v + 80)) / 6.8))"
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
        "\n"
        "\n    # Assign states"
        "\n    v = states[0]"
        "\n"
        "\n    # Assign parameters"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    dv_dt = 0"
        "\n    values[0] = dv_dt"
        "\n    tm = 0.06487 * numpy.exp(-(((v - 4.823) / 51.12) ** 2))"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_remove_unused_rhs(ode_unused):
    codegen_orig = PythonCodeGenerator(ode_unused)
    assert codegen_orig.rhs() == (
        "def rhs(t, states, parameters):"
        "\n"
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
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
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
    codegen_remove = PythonCodeGenerator(ode_unused, remove_unused=True)
    assert codegen_remove.rhs() == (
        "def rhs(t, states, parameters):"
        "\n"
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
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
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


def test_python_remove_unused_forward_explicit_euler(ode_unused):
    codegen_orig = PythonCodeGenerator(ode_unused)
    assert codegen_orig.scheme(get_scheme("forward_explicit_euler")) == (
        "def forward_explicit_euler(states, t, dt, parameters):"
        "\n"
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
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
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
    codegen_remove = PythonCodeGenerator(ode_unused, remove_unused=True)
    assert codegen_remove.scheme(get_scheme("forward_explicit_euler")) == (
        "def forward_explicit_euler(states, t, dt, parameters):"
        "\n"
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
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
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


def test_python_remove_unused_forward_generalized_rush_larsen(ode_unused):
    codegen_orig = PythonCodeGenerator(ode_unused)
    assert codegen_orig.scheme(get_scheme("forward_generalized_rush_larsen")) == (
        "def forward_generalized_rush_larsen(states, t, dt, parameters):"
        "\n"
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
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    unused_expression = 0"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dt * dunused_state_dt + unused_state"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    dx_dt_linearized = -sigma"
        "\n    values[1] = x + numpy.where("
        "\n        numpy.logical_or((dx_dt_linearized > 1e-08), (dx_dt_linearized < -1e-08)),"
        "\n        dx_dt * (numpy.exp(dt * dx_dt_linearized) - 1) / dx_dt_linearized,"
        "\n        dt * dx_dt,"
        "\n    )"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    dy_dt_linearized = -1"
        "\n    values[2] = y + numpy.where("
        "\n        numpy.logical_or((dy_dt_linearized > 1e-08), (dy_dt_linearized < -1e-08)),"
        "\n        dy_dt * (numpy.exp(dt * dy_dt_linearized) - 1) / dy_dt_linearized,"
        "\n        dt * dy_dt,"
        "\n    )"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    dz_dt_linearized = -beta"
        "\n    values[3] = z + numpy.where("
        "\n        numpy.logical_or((dz_dt_linearized > 1e-08), (dz_dt_linearized < -1e-08)),"
        "\n        dz_dt * (numpy.exp(dt * dz_dt_linearized) - 1) / dz_dt_linearized,"
        "\n        dt * dz_dt,"
        "\n    )"
        "\n"
        "\n    return values"
        "\n"
    )
    codegen_remove = PythonCodeGenerator(ode_unused, remove_unused=True)
    assert codegen_remove.scheme(get_scheme("forward_generalized_rush_larsen")) == (
        "def forward_generalized_rush_larsen(states, t, dt, parameters):"
        "\n"
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
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    dunused_state_dt = 0"
        "\n    values[0] = dt * dunused_state_dt + unused_state"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    dx_dt_linearized = -sigma"
        "\n    values[1] = x + numpy.where("
        "\n        numpy.logical_or((dx_dt_linearized > 1e-08), (dx_dt_linearized < -1e-08)),"
        "\n        dx_dt * (numpy.exp(dt * dx_dt_linearized) - 1) / dx_dt_linearized,"
        "\n        dt * dx_dt,"
        "\n    )"
        "\n    dy_dt = x * (rho - z) - y"
        "\n    dy_dt_linearized = -1"
        "\n    values[2] = y + numpy.where("
        "\n        numpy.logical_or((dy_dt_linearized > 1e-08), (dy_dt_linearized < -1e-08)),"
        "\n        dy_dt * (numpy.exp(dt * dy_dt_linearized) - 1) / dy_dt_linearized,"
        "\n        dt * dy_dt,"
        "\n    )"
        "\n    dz_dt = (-beta) * z + x * y"
        "\n    dz_dt_linearized = -beta"
        "\n    values[3] = z + numpy.where("
        "\n        numpy.logical_or((dz_dt_linearized > 1e-08), (dz_dt_linearized < -1e-08)),"
        "\n        dz_dt * (numpy.exp(dt * dz_dt_linearized) - 1) / dz_dt_linearized,"
        "\n        dt * dz_dt,"
        "\n    )"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_monitored(codegen: PythonCodeGenerator):
    assert codegen.monitor_values() == (
        "def monitor_values(t, states, parameters):"
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    z = states[1]"
        "\n    y = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    shape = 5 if len(states.shape) == 1 else (5, states.shape[1])"
        "\n    values = numpy.zeros(shape)"
        "\n    betaz = beta * z"
        "\n    values[0] = betaz"
        "\n    rhoz = rho - z"
        "\n    values[1] = rhoz"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    values[2] = dx_dt"
        "\n    dz_dt = -betaz + x * y"
        "\n    values[3] = dz_dt"
        "\n    dy_dt = rhoz * x - y"
        "\n    values[4] = dy_dt"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_monitored_index(codegen: PythonCodeGenerator):
    assert codegen.monitor_index() == (
        'monitor = {"betaz": 0, "rhoz": 1, "dx_dt": 2, "dz_dt": 3, "dy_dt": 4}'
        "\n"
        "\n"
        "\ndef monitor_index(name: str) -> int:"
        '\n    """Return the index of the monitor with the given name'
        "\n"
        "\n    Arguments"
        "\n    ---------"
        "\n    name : str"
        "\n        The name of the monitor"
        "\n"
        "\n    Returns"
        "\n    -------"
        "\n    int"
        "\n        The index of the monitor"
        "\n"
        "\n    Raises"
        "\n    ------"
        "\n    KeyError"
        "\n        If the name is not a valid monitor"
        '\n    """'
        "\n"
        "\n    return monitor[name]"
        "\n"
    )


@pytest.fixture(scope="module")
def singular_ode(parser, trans):
    expr = """
    parameters(
    a = 1.0,
    b = 2.0
    )
    states(
    x = 1.0
    )

    y = x /(exp(x) - 1.0)
    z = x /(exp(x) - 1.0) + (x - 2) /(exp(x) - exp(2))
    dx_dt = b / x
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree), name="lorentz")


def test_codegen_rhs_singular_ode(singular_ode):
    codegen_sing = PythonCodeGenerator(singular_ode)

    assert codegen_sing.rhs() == (
        "def rhs(t, states, parameters):"
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    b = parameters[1]"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    y = x / (numpy.exp(x) - 1.0)"
        "\n    z = x / (numpy.exp(x) - 1.0) + (x - 2) / (numpy.exp(x) - numpy.exp(2))"
        "\n    dx_dt = b / x"
        "\n    values[0] = dx_dt"
        "\n"
        "\n    return values"
        "\n"
    )

    new_ode = singular_ode.remove_singularities()
    codegen_fixed = PythonCodeGenerator(new_ode)
    assert codegen_fixed.rhs() == (
        "def rhs(t, states, parameters):"
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    b = parameters[1]"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    y = numpy.where((x == 0), 1, x / (numpy.exp(x) - 1.0))"
        "\n    z = numpy.where("
        "\n        numpy.logical_and((x == 0), (x == 2)),"
        "\n        numpy.exp(-2) - 2 / (1 - numpy.exp(2)) + 2 / (-1 + numpy.exp(2)) + 1,"
        "\n        numpy.where("
        "\n            (x == 0),"
        "\n            x / (numpy.exp(x) - 1.0)"
        "\n            + (x - 2) / (numpy.exp(x) - numpy.exp(2))"
        "\n            - 2 / (1 - numpy.exp(2))"
        "\n            + 1,"
        "\n            numpy.where("
        "\n                (x == 2),"
        "\n                x / (numpy.exp(x) - 1.0)"
        "\n                + (x - 2) / (numpy.exp(x) - numpy.exp(2))"
        "\n                + numpy.exp(-2)"
        "\n                + 2 / (-1 + numpy.exp(2)),"
        "\n                2 * x / (numpy.exp(x) - 1.0)"
        "\n                + 2 * (x - 2) / (numpy.exp(x) - numpy.exp(2)),"
        "\n            ),"
        "\n        ),"
        "\n    )"
        "\n    dx_dt = b / x"
        "\n    values[0] = dx_dt"
        "\n"
        "\n    return values"
        "\n"
    )


def test_python_jax_codegen_initial_state_values(codegen_jax: JaxCodeGenerator):
    assert codegen_jax.initial_state_values() == (
        "@jax.jit"
        "\ndef init_state_values(**values):"
        '\n    """Initialize state values"""'
        "\n    # x=1.0, z=3.05, y=2.0"
        "\n"
        "\n    states = numpy.array([1.0, 3.05, 2.0], dtype=numpy.float64)"
        "\n"
        "\n    for key, value in values.items():"
        "\n        states = states.at[state_index(key)].set(value)"
        "\n"
        "\n    return states"
        "\n"
    )


def test_python_jax_codegen_rhs(codegen_jax: JaxCodeGenerator):
    assert codegen_jax.rhs() == (
        "@jax.jit"
        "\ndef rhs(t, states, parameters):"
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n    z = states[1]"
        "\n    y = states[2]"
        "\n"
        "\n    # Assign parameters"
        "\n    a = parameters[0]"
        "\n    beta = parameters[1]"
        "\n    rho = parameters[2]"
        "\n    sigma = parameters[3]"
        "\n"
        "\n    # Assign expressions"
        "\n    betaz = beta * z"
        "\n    rhoz = rho - z"
        "\n    dx_dt = sigma * (-x + y)"
        "\n    _values_0 = dx_dt"
        "\n    dz_dt = -betaz + x * y"
        "\n    _values_1 = dz_dt"
        "\n    dy_dt = rhoz * x - y"
        "\n    _values_2 = dy_dt"
        "\n"
        "\n    return numpy.array("
        "\n        ["
        "\n            _values_0,"
        "\n            _values_1,"
        "\n            _values_2,"
        "\n        ]"
        "\n    )"
        "\n"
    )


def test_python_jax_codegen_initial_parameter_values(codegen_jax: JaxCodeGenerator):
    assert codegen_jax.initial_parameter_values() == (
        "@jax.jit"
        "\ndef init_parameter_values(**values):"
        '\n    """Initialize parameter values"""'
        "\n    # a=0, beta=2.4, rho=21.0, sigma=12.0"
        "\n"
        "\n    parameters = numpy.array([0, 2.4, 21.0, 12.0], dtype=numpy.float64)"
        "\n"
        "\n    for key, value in values.items():"
        "\n        parameters = parameters.at[parameter_index(key)].set(value)"
        "\n"
        "\n    return parameters"
        "\n"
    )


def test_conditional_times_float(parser, trans):
    expr = """ \
    \nstates(x=0)
    \ndx_dt = (-1.0 - x) * Conditional(Lt(x, -1.0), 1.0,0.0) \
    """

    tree = parser.parse(expr)
    result = trans.transform(tree)
    ode = make_ode(*result, name="name")
    codegen = PythonCodeGenerator(ode)
    rhs = codegen.rhs()
    assert rhs == (
        "def rhs(t, states, parameters):"
        "\n"
        "\n    # Assign states"
        "\n    x = states[0]"
        "\n"
        "\n    # Assign parameters"
        "\n"
        "\n    # Assign expressions"
        "\n"
        "\n    values = numpy.zeros_like(states, dtype=numpy.float64)"
        "\n    dx_dt = (-x - 1.0) * numpy.where((x < -1.0), 1.0, 0.0)"
        "\n    values[0] = dx_dt"
        "\n"
        "\n    return values"
        "\n"
    )
