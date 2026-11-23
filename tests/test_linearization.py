import sympy

import gotranx
from gotranx.linearization import diagonal_jacobian, gating_states
from gotranx.ode import make_ode


def build(parser, trans, expr):
    return make_ode(*trans.transform(parser.parse(expr)))


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


def test_differentiates_through_an_intermediate(parser, trans):
    """The bug: d(dV_dt)/dV is -g, not 0, even when written via I."""
    ode = build(parser, trans, VIA_INTERMEDIATE)
    (g,) = [p.symbol for p in ode.parameters if p.name == "g"]
    assert sympy.simplify(diagonal_jacobian(ode)["V"] + g) == 0


def test_is_invariant_to_naming_a_subexpression(parser, trans):
    """The property the method requires: f does not depend on how it is spelled."""
    inlined = diagonal_jacobian(build(parser, trans, INLINED))["V"]
    via = diagonal_jacobian(build(parser, trans, VIA_INTERMEDIATE))["V"]
    assert sympy.simplify(inlined - via) == 0


def test_chains_through_two_levels_of_intermediate(parser, trans):
    ode = build(
        parser,
        trans,
        """
        parameters(a=2.0, b=3.0)
        states("C", x=1.0)

        expressions("C")
        u = a*x
        w = b*u
        dx_dt = -w
        """,
    )
    (a,) = [p.symbol for p in ode.parameters if p.name == "a"]
    (b,) = [p.symbol for p in ode.parameters if p.name == "b"]
    # dx_dt = -b*a*x  =>  d/dx = -a*b
    assert sympy.simplify(diagonal_jacobian(ode)["x"] + a * b) == 0


def test_is_zero_when_the_derivative_does_not_depend_on_its_own_state(parser, trans):
    """Not a failure: forward Euler is the exact g->0 limit of the RL update."""
    ode = build(
        parser,
        trans,
        """
        states("C", x=1.0, y=2.0)

        expressions("C")
        dx_dt = y
        dy_dt = 0
        """,
    )
    jac = diagonal_jacobian(ode)
    assert jac["x"].is_zero
    assert jac["y"].is_zero


def test_covers_every_state(parser, trans):
    ode = build(
        parser,
        trans,
        """
        parameters(a=1.0)
        states("C", x=1.0, y=2.0, z=3.0)

        expressions("C")
        p = a*y
        dx_dt = -x
        dy_dt = -p
        dz_dt = -z*y
        """,
    )
    assert set(diagonal_jacobian(ode)) == {"x", "y", "z"}


def test_does_not_confuse_symbols_that_share_a_name(parser, trans):
    """gotranx symbols carry real=True; a plain Symbol must not match."""
    ode = build(parser, trans, VIA_INTERMEDIATE)
    result = diagonal_jacobian(ode)["V"]
    assert sympy.Symbol("g") not in result.free_symbols
    assert sympy.Symbol("g", real=True) in result.free_symbols


def _ode(text):
    return gotranx.load.ode_from_string(text)


def test_gating_states_detects_hodgkin_huxley_gates_only():
    ode = _ode(
        """
        parameters(g=1.0)
        states(V=-80.0, m=0.1, h=0.6)
        minf = 1/(1 + exp(-(V + 40)/7))
        tau_m = 0.1 + 0.5*exp(-((V + 40)/30)**2)
        alpha_h = 0.07*exp(-(V + 65)/20)
        beta_h = 1/(1 + exp(-(V + 35)/10))
        dm_dt = (minf - m)/tau_m
        dh_dt = alpha_h*(1 - h) - beta_h*h
        dV_dt = -g*m**3*h*(V - 50) - 0.1*(V + 80)**3/1000
        """
    )
    # ode.states is sorted by name, so compare sets
    assert set(gating_states(ode)) == {"m", "h"}


def test_gating_states_rejects_a_state_nonlinear_in_itself():
    ode = _ode("states(y=1.0)\ndy_dt = -y**2\n")
    assert gating_states(ode) == []


def test_gating_states_rejects_a_conditional_on_the_state_itself():
    ode = _ode("states(y=1.0)\ndy_dt = Conditional(Gt(y, 0.5), -y, -2*y)\n")
    assert gating_states(ode) == []


def test_gating_states_rejects_a_conditional_on_the_state_through_an_intermediate():
    ode = _ode("states(y=1.0)\nrate = Conditional(Gt(y, 0.5), 1.0, 2.0)\ndy_dt = -rate*y\n")
    assert gating_states(ode) == []


def test_gating_states_accepts_a_conditional_on_another_state():
    ode = _ode(
        "states(V=0.0, y=1.0)\nrate = Conditional(Gt(V, 0.5), 1.0, 2.0)\n"
        "dy_dt = -rate*y\ndV_dt = -V\n"
    )
    assert set(gating_states(ode)) == {"V", "y"}


def test_gating_states_keeps_only_states_whose_derivative_depends_on_them():
    ode = _ode("states(x=1.0, y=0.0)\ndx_dt = -x\ndy_dt = x\n")
    assert gating_states(ode) == ["x"]


def test_gating_states_is_exported():
    import gotranx.linearization as lin

    assert "gating_states" in lin.__all__
