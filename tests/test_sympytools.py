import pytest
import sympy
from gotranx.ode import make_ode
from gotranx.ode import ODE
from gotranx import sympytools


@pytest.fixture(scope="module")
def ode(trans, parser) -> ODE:
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4
    )
    states("My component", x=1.0, z=3.05, y=2.0)

    expressions("My component")
    dx_dt = sigma*(-x + y)
    dy_dt = y_int - y # millivolt
    dz_dt = z_int + x*y
    y_int = x*(rho - z)
    z_int = -beta*z
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree))


# @pytest.fixture(scope="module")
# def sym_ode(ode: ODE) -> SympyODE:
#     return SympyODE(ode)


def test_states(ode):
    states = sympytools.states_matrix(ode)
    assert len(states) == 3
    assert str(states[0]) == "x"
    assert str(states[1]) == "y"
    assert str(states[2]) == "z"


def test_Conditional_boolean_condition():
    assert sympytools.Conditional(True, 1, 2) == 1


def test_Conditional_invalid_condition():
    with pytest.raises(TypeError):
        sympytools.Conditional(sympy.Eq, 1, 2)


def test_ContinuousConditional_invalid_condition():
    with pytest.raises(TypeError):
        sympytools.ContinuousConditional(sympy.Eq, 1, 2)


def test_rhs_matrix(ode: ODE):
    rhs = sympytools.rhs_matrix(ode)
    assert len(rhs) == 3
    assert str(rhs[0]) == "sigma*(-x + y)"
    assert str(rhs[1]) == "x*(rho - z) - y"
    assert str(rhs[2]) == "(-beta)*z + x*y"


def test_jacobian_matrix(ode: ODE):
    jac = sympytools.jacobi_matrix(ode)

    assert str(jac[0]) == "-sigma"
    assert str(jac[1]) == "sigma"
    assert str(jac[2]) == "0"

    assert str(jac[3]) == "rho - z"
    assert str(jac[4]) == "-1"
    assert str(jac[5]) == "-x"

    assert str(jac[6]) == "y"
    assert str(jac[7]) == "x"
    assert str(jac[8]) == "-beta"
