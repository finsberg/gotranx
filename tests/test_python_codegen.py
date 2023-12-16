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
    states("My component", "info about states", x=1.0, y=2.0,z=3.05)

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
        "\n    states = np.array([1.0, 2.0, 3.05])"
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
        "\n    # a=0.0, beta=2.4, rho=21.0, sigma=12.0"
        "\n"
        "\n    parameters = np.array([0.0, 2.4, 21.0, 12.0])"
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
        "\n    values = np.zeros(3)"
        "\n    values[0] = sigma * (-x + y)"
        "\n    values[1] = x * (rho - z) - y"
        "\n    values[2] = -beta * z + x * y"
        "\n"
        "\n    return values"
        "\n"
    )
