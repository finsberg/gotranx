import pytest
import sympy
from gotranx.ode import make_ode
from gotranx.ode import ODE
from gotranx import schemes


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


def test_forward_explicit_euler(ode: ODE):
    dt = sympy.Symbol("dt")
    eqs = schemes.forward_explicit_euler(ode, dt)
    assert len(eqs) == 8

    assert str(eqs[0]) == "Eq(y_int, x*(rho - z))"
    assert str(eqs[1]) == "Eq(z_int, -beta*z)"
    assert str(eqs[2]) == "Eq(dx_dt, sigma*(-x + y))"
    assert str(eqs[3]) == "Eq(values[0], dt*dx_dt + x)"
    assert str(eqs[4]) == "Eq(dy_dt, -y + y_int)"
    assert str(eqs[5]) == "Eq(values[1], dt*dy_dt + y)"
    assert str(eqs[6]) == "Eq(dz_dt, x*y + z_int)"
    assert str(eqs[7]) == "Eq(values[2], dt*dz_dt + z)"


def test_forward_generalized_rush_larsen(ode: ODE):
    dt = sympy.Symbol("dt")
    eqs = schemes.forward_generalized_rush_larsen(ode, dt)

    assert len(eqs) == 10

    assert str(eqs[0]) == "Eq(y_int, x*(rho - z))"
    assert str(eqs[1]) == "Eq(z_int, -beta*z)"
    assert str(eqs[2]) == "Eq(dx_dt, sigma*(-x + y))"
    assert str(eqs[3]) == "Eq(dx_dt_linearized, -sigma)"
    assert str(eqs[4]) == (
        "Eq(values[0], x + "
        "Piecewise((dx_dt*(exp(dt*dx_dt_linearized) - 1)"
        "/dx_dt_linearized, Abs(dx_dt_linearized) > 1.0e-8), "
        "(dt*dx_dt, True)))"
    )
    assert str(eqs[5]) == "Eq(dy_dt, -y + y_int)"
    assert str(eqs[6]) == "Eq(dy_dt_linearized, -1)"
    assert str(eqs[7]) == (
        "Eq(values[1], y + "
        "Piecewise((dy_dt*(exp(dt*dy_dt_linearized) - 1)"
        "/dy_dt_linearized, Abs(dy_dt_linearized) > 1.0e-8), "
        "(dt*dy_dt, True)))"
    )
