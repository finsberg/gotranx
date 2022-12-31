import pytest
from gotranx.ode import make_ode
from gotranx.ode import ODE
from gotranx.sympy_ode import SympyODE


@pytest.fixture(scope="module")
def ode(trans, parser) -> ODE:
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", "info about states", x=1.0, z=3.05, y=2.0)

    expressions("My component")
    dx_dt = sigma*(-x + y)
    dy_dt = y_int - y # millivolt
    dz_dt = z_int + x*y
    y_int = x*(rho - z)
    z_int = -beta*z
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree))


@pytest.fixture(scope="module")
def sym_ode(ode: ODE) -> SympyODE:
    return SympyODE(ode)


def test_states(sym_ode):
    assert len(sym_ode.states) == 3
    assert str(sym_ode.states[0]) == "x"
    assert str(sym_ode.states[1]) == "y"
    assert str(sym_ode.states[2]) == "z"


def test_state_values(sym_ode):
    assert len(sym_ode.states) == 3
    assert str(sym_ode.states[0]) == "x"
    assert str(sym_ode.states[1]) == "y"
    assert str(sym_ode.states[2]) == "z"


def test_parameters(sym_ode):
    assert len(sym_ode.parameters) == 4
    assert str(sym_ode.parameters[0]) == "a"
    assert str(sym_ode.parameters[1]) == "beta"
    assert str(sym_ode.parameters[2]) == "rho"
    assert str(sym_ode.parameters[3]) == "sigma"


def test_state_derivatives(sym_ode):
    assert len(sym_ode.state_derivatives) == 3
    assert str(sym_ode.state_derivatives[0]) == "dx_dt"
    assert str(sym_ode.state_derivatives[1]) == "dy_dt"
    assert str(sym_ode.state_derivatives[2]) == "dz_dt"


def test_rhs(sym_ode):
    assert len(sym_ode.rhs) == 3
    assert str(sym_ode.rhs[0]) == "sigma*(-x + y)"
    assert str(sym_ode.rhs[1]) == "x*(rho - z) - y"
    assert str(sym_ode.rhs[2]) == "-beta*z + x*y"


def test_jacobian(sym_ode):
    jac = sym_ode.jacobian
    assert str(jac[0]) == "-sigma"
    assert str(jac[1]) == "sigma"
    assert str(jac[2]) == "0"

    assert str(jac[3]) == "rho - z"
    assert str(jac[4]) == "-1"
    assert str(jac[5]) == "-x"

    assert str(jac[6]) == "y"
    assert str(jac[7]) == "x"
    assert str(jac[8]) == "-beta"
