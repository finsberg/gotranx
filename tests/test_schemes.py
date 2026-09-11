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


def test_explicit_euler(ode: ODE):
    dt = sympy.Symbol("dt")
    eqs = schemes.explicit_euler(ode, dt)
    assert len(eqs) == 8

    assert eqs[0] == "y_int = x*(rho - z)"
    assert eqs[1] == "z_int = (-beta)*z"
    assert eqs[2] == "dx_dt = sigma*(-x + y)"
    assert eqs[3] == "values[0] = dt*dx_dt + x"
    assert eqs[4] == "dy_dt = -y + y_int"
    assert eqs[5] == "values[1] = dt*dy_dt + y"
    assert eqs[6] == "dz_dt = x*y + z_int"
    assert eqs[7] == "values[2] = dt*dz_dt + z"


def test_generalized_rush_larsen(ode: ODE):
    dt = sympy.Symbol("dt")
    eqs = schemes.generalized_rush_larsen(ode, dt)

    assert len(eqs) == 11

    assert str(eqs[0]) == "y_int = x*(rho - z)"
    assert str(eqs[1]) == "z_int = (-beta)*z"
    assert str(eqs[2]) == "dx_dt = sigma*(-x + y)"
    assert str(eqs[3]) == "dx_dt_linearized = -sigma"
    assert str(eqs[4]) == (
        "values[0] = x + "
        "((dx_dt*(math.exp(dt*dx_dt_linearized) - 1)"
        "/dx_dt_linearized) if (abs(dx_dt_linearized) > 1.0e-8) "
        "else (dt*dx_dt))"
    )
    assert str(eqs[5]) == "dy_dt = -y + y_int"
    assert str(eqs[6]) == "dy_dt_linearized = -1"
    assert str(eqs[7]) == (
        "values[1] = y + "
        "((dy_dt*(math.exp(dt*dy_dt_linearized) - 1)"
        "/dy_dt_linearized) if (abs(dy_dt_linearized) > 1.0e-8) "
        "else (dt*dy_dt))"
    )
    assert str(eqs[8]) == "dz_dt = x*y + z_int"
    assert str(eqs[9]) == "dz_dt_linearized = -beta"
    assert str(eqs[10]) == (
        "values[2] = z + "
        "((dz_dt*(math.exp(dt*dz_dt_linearized) - 1)"
        "/dz_dt_linearized) if (abs(dz_dt_linearized) > 1.0e-8) "
        "else (dt*dz_dt))"
    )


def test_hybrid_rush_larsen(ode: ODE):
    dt = sympy.Symbol("dt")
    eqs = schemes.hybrid_rush_larsen(ode, dt, stiff_states=["y", "z"])

    assert len(eqs) == 9

    assert str(eqs[0]) == "y_int = x*(rho - z)"
    assert str(eqs[1]) == "z_int = (-beta)*z"
    assert str(eqs[2]) == "dx_dt = sigma*(-x + y)"
    assert str(eqs[3]) == "values[0] = dt*dx_dt + x"
    assert str(eqs[4]) == "dy_dt = -y + y_int"
    assert str(eqs[5]) == "dy_dt_linearized = -1"
    assert str(eqs[6]) == (
        "values[1] = y + "
        "((dy_dt*(math.exp(dt*dy_dt_linearized) - 1)"
        "/dy_dt_linearized) if (abs(dy_dt_linearized) > 1.0e-8) "
        "else (dt*dy_dt))"
    )
    assert str(eqs[7]) == "dz_dt = x*y + z_int"
    assert str(eqs[8]) == "values[2] = dt*dz_dt + z"


INLINED = """
parameters(g=100.0, E=-85.0)
states("Membrane", V=-80.0)

expressions("Membrane")
dV_dt = -(g*(V - E))
"""

VIA_INTERMEDIATE = """
parameters(g=100.0, E=-85.0)
states("Membrane", V=-80.0)

expressions("Membrane")
I = g*(V - E)
dV_dt = -I
"""


def _invariance_case(parser, trans, scheme, **scheme_kwargs):
    """Generate both spellings and return the lines that encode the scheme."""
    dt = sympy.Symbol("dt")
    out = []
    for expr in (INLINED, VIA_INTERMEDIATE):
        ode = make_ode(*trans.transform(parser.parse(expr)))
        eqs = [str(e) for e in scheme(ode, dt, **scheme_kwargs)]
        out.append([e for e in eqs if "linearized" in e or e.startswith("values[")])
    return out


def test_generalized_rush_larsen_is_invariant_to_naming_a_subexpression(parser, trans):
    """Naming a current must not change the scheme. Both spellings describe
    the same f, so both must linearize to -g."""
    inlined, via_intermediate = _invariance_case(parser, trans, schemes.generalized_rush_larsen)
    assert inlined == via_intermediate
    assert "dV_dt_linearized = -g" in inlined
