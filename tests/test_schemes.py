import re

import pytest
import sympy
from structlog.testing import capture_logs
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

    assert len(eqs) == 10

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
    assert str(eqs[8]) == "dz_dt_linearized = -beta"
    assert str(eqs[9]) == (
        "values[2] = z + "
        "((dz_dt*(math.exp(dt*dz_dt_linearized) - 1)"
        "/dz_dt_linearized) if (abs(dz_dt_linearized) > 1.0e-8) "
        "else (dt*dz_dt))"
    )


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


def test_generalized_rush_larsen_emits_shared_subexpression_as_temporary(parser, trans):
    """A Jacobian diagonal entry with a genuine repeated subexpression must be
    factored by per-state CSE.

    ``d/dx[sin(x**2)*cos(x**2)]`` is ``2*x*cos(x**2)**2 - 2*x*sin(x**2)**2``,
    in which ``x**2`` occurs four times; confirmed with a standalone
    ``sympy.cse`` call before writing this test. Every Jacobian entry reached
    by the other scheme tests is a single symbol, so none of them exercise the
    temporary-emission loop or the `fresh()` name generator in
    `_linearized_assignments` -- this test is the one that does.

    Passes ``cse="per_state"`` explicitly: this test's whole point is the
    per-state naming (`_dx_dt_linearized_N`) produced by
    `_linearized_assignments`, which is no longer what the default strategy
    (joint) does -- joint uses the shared `_linearization_temp_N` pool
    instead. See `test_generalized_rush_larsen_joint_cse_shares_temporary_across_states`
    for the joint-naming equivalent.
    """
    expr = """
    states(x=0.5)
    dx_dt = sin(x**2)*cos(x**2)
    """
    ode = make_ode(*trans.transform(parser.parse(expr)))
    dt = sympy.Symbol("dt")
    eqs = [str(e) for e in schemes.generalized_rush_larsen(ode, dt, cse="per_state")]

    temp_indices = [i for i, e in enumerate(eqs) if e.startswith("_dx_dt_linearized_")]
    assert temp_indices, f"expected at least one CSE temporary, got none: eqs={eqs}"

    state_derivative_index = next(i for i, e in enumerate(eqs) if e.startswith("dx_dt = "))
    linearized_index = next(i for i, e in enumerate(eqs) if e.startswith("dx_dt_linearized = "))

    # Per-state CSE is only safe to emit inline if it comes after the state
    # derivative it was factored out of, and before the `_linearized` line
    # that consumes it.
    assert state_derivative_index < min(temp_indices)
    assert max(temp_indices) < linearized_index

    temp_names = [eqs[i].split(" = ")[0] for i in temp_indices]
    assert any(temp_name in eqs[linearized_index] for temp_name in temp_names)


def test_linearized_assignments_skips_taken_names():
    """`_linearized_assignments`'s `fresh()` generator must skip names already
    in `taken`, not merely start counting past them.

    `sympy.cse(..., symbols=...)` consumes the generator lazily, one name per
    membership check, so "start past the taken count" and "skip taken names"
    only coincide when `taken` is contiguous from `_0`. Reusing the same
    repeated-subexpression case as
    `test_generalized_rush_larsen_emits_shared_subexpression_as_temporary`,
    but calling the helper directly and pre-seeding `_dx_dt_linearized_0` as
    taken.
    """
    x = sympy.Symbol("x")
    expr = -2 * x * sympy.sin(x**2) ** 2 + 2 * x * sympy.cos(x**2) ** 2

    replacements, _ = schemes._linearized_assignments("dx_dt", expr, taken={"_dx_dt_linearized_0"})

    assert replacements
    assert replacements[0][0].name == "_dx_dt_linearized_1"


def test_generalized_rush_larsen_default_cse_is_joint():
    """Joint is the new default: it costs the same or fewer operations than
    per-state on every backend measured (see schemes.py background), so it
    should not need to be opted into."""
    import inspect

    sig = inspect.signature(schemes.generalized_rush_larsen)
    assert sig.parameters["cse"].default == schemes.CSEStrategy.joint


def test_hybrid_rush_larsen_default_cse_is_joint():
    import inspect

    sig = inspect.signature(schemes.hybrid_rush_larsen)
    assert sig.parameters["cse"].default == schemes.CSEStrategy.joint


def test_cse_strategy_accepts_plain_strings(ode):
    """Schemes are invoked with `**kwargs` from `CodeGenerator.scheme`, so a
    plain string (e.g. from a config file) must work exactly like the enum
    member."""
    dt = sympy.Symbol("dt")
    from_enum = schemes.generalized_rush_larsen(ode, dt, cse=schemes.CSEStrategy.none)
    from_string = schemes.generalized_rush_larsen(ode, dt, cse="none")
    assert [str(e) for e in from_enum] == [str(e) for e in from_string]


def test_invalid_cse_strategy_raises_a_clear_error(ode):
    dt = sympy.Symbol("dt")
    with pytest.raises(ValueError, match="not-a-real-strategy"):
        schemes.generalized_rush_larsen(ode, dt, cse="not-a-real-strategy")


def test_cse_none_inlines_the_full_expression_with_no_temporaries(parser, trans):
    """`cse="none"` must not factor anything: `d<state>_dt_linearized` is one
    inlined expression, with no `_linearization_temp_*` or per-state
    temporaries at all."""
    expr = """
    states(x=0.5)
    dx_dt = sin(x**2)*cos(x**2)
    """
    ode = make_ode(*trans.transform(parser.parse(expr)))
    dt = sympy.Symbol("dt")
    eqs = [str(e) for e in schemes.generalized_rush_larsen(ode, dt, cse="none")]

    assert not [e for e in eqs if e.startswith("_dx_dt_linearized_")]
    assert not [e for e in eqs if e.startswith("_linearization_temp_")]
    linearized = next(e for e in eqs if e.startswith("dx_dt_linearized = "))
    # The raw, un-factored diagonal Jacobian entry.
    assert linearized == "dx_dt_linearized = -2*x*math.sin(x**2)**2 + 2*x*math.cos(x**2)**2"


JOINT_SHARED_SUBEXPRESSION = """
states(x=1.0, y=1.0)
q = sin(x + y)*cos(x + y)
dx_dt = q - x
dy_dt = q - y
"""


def test_generalized_rush_larsen_joint_cse_shares_temporary_across_states(parser, trans):
    """The crux of joint CSE: a subexpression shared *between* two states'
    linearized derivatives must be computed once, as a `_linearization_temp_*`
    temporary emitted immediately before the first state that needs it (here,
    x), and simply reused (not recomputed) for the second (y).

    `d(dx_dt)/dx` and `d(dy_dt)/dy` are both `cos(x+y)**2 - sin(x+y)**2 - 1`
    here (since `d(x+y)/dx == d(x+y)/dy == 1`), confirmed with a standalone
    `sympy.cse` call before writing this test: `sympy.cse([jac_x, jac_y])`
    returns one temporary for `x+y` and a second temporary, identical for
    both states, for the whole reduced expression.
    """
    ode = make_ode(*trans.transform(parser.parse(JOINT_SHARED_SUBEXPRESSION)))
    dt = sympy.Symbol("dt")
    eqs = [str(e) for e in schemes.generalized_rush_larsen(ode, dt)]  # default: joint

    temp_indices = [i for i, e in enumerate(eqs) if e.startswith("_linearization_temp_")]
    assert temp_indices, f"expected joint CSE temporaries, got none: eqs={eqs}"

    dx_linearized_index = next(i for i, e in enumerate(eqs) if e.startswith("dx_dt_linearized = "))
    dy_linearized_index = next(i for i, e in enumerate(eqs) if e.startswith("dy_dt_linearized = "))

    # Every temporary is emitted before the first (x's) linearized line, and
    # none are re-emitted before the second (y's).
    assert all(i < dx_linearized_index for i in temp_indices)
    between = eqs[dx_linearized_index + 1 : dy_linearized_index]
    assert not [e for e in between if e.split(" = ")[0].startswith("_linearization_temp_")]

    # Both linearized lines resolve to (possibly the same) temporary, not to
    # two independently-factored copies.
    dx_line = eqs[dx_linearized_index]
    dy_line = eqs[dy_linearized_index]
    assert dx_line.split(" = ", 1)[1] == dy_line.split(" = ", 1)[1]


def test_every_temporary_is_emitted_before_its_first_use(parser, trans):
    """No generated line may reference a name that has not appeared as the
    left-hand side of an earlier line in the same list -- this would be a
    `NameError` at run time in the generated code."""
    ode = make_ode(*trans.transform(parser.parse(JOINT_SHARED_SUBEXPRESSION)))
    dt = sympy.Symbol("dt")
    eqs = [str(e) for e in schemes.generalized_rush_larsen(ode, dt)]

    lhs_names = {e.split(" = ", 1)[0].strip() for e in eqs}
    defined: set[str] = set()
    for e in eqs:
        lhs, rhs = e.split(" = ", 1)
        lhs = lhs.strip()
        used = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", rhs))
        forward_refs = (used & lhs_names) - defined
        assert not forward_refs, f"{lhs!r} uses {forward_refs} before it is defined: eqs={eqs}"
        defined.add(lhs)


def test_hybrid_rush_larsen_does_not_claim_a_present_state_is_missing(parser, trans):
    """A stiff state that is in the ODE must never be reported as not found."""
    ode = make_ode(*trans.transform(parser.parse(VIA_INTERMEDIATE)))
    with capture_logs() as logs:
        schemes.hybrid_rush_larsen(ode, sympy.Symbol("dt"), stiff_states=["V"])

    assert not [log for log in logs if "not found in the ODE" in log.get("event", "")]


def test_hybrid_rush_larsen_warns_about_a_stiff_state_that_is_not_in_the_ode(parser, trans):
    """A genuine typo in --stiff-states should be audible."""
    ode = make_ode(*trans.transform(parser.parse(VIA_INTERMEDIATE)))
    with capture_logs() as logs:
        schemes.hybrid_rush_larsen(ode, sympy.Symbol("dt"), stiff_states=["Vm"])

    warnings = [log for log in logs if log.get("log_level") == "warning"]
    assert any("not found in the ODE" in log["event"] for log in warnings)
    assert any("Vm" in log["event"] for log in warnings)


def test_hybrid_rush_larsen_warns_when_a_stiff_state_cannot_be_linearized(parser, trans):
    """A user asking for RL on a state with zero df/dx gets forward Euler.
    That request cannot be honored, so say so."""
    ode = make_ode(
        *trans.transform(
            parser.parse(
                """
                states("C", x=1.0, y=2.0)

                expressions("C")
                dx_dt = y
                dy_dt = 0
                """
            )
        )
    )
    with capture_logs() as logs:
        schemes.hybrid_rush_larsen(ode, sympy.Symbol("dt"), stiff_states=["x"])

    warnings = [log for log in logs if log.get("log_level") == "warning"]
    assert any("x" in log["event"] for log in warnings)


def test_hybrid_rush_larsen_is_invariant_to_naming_a_subexpression(parser, trans):
    """The Task 2 property, for the hybrid scheme with V marked stiff."""
    inlined, via_intermediate = _invariance_case(
        parser, trans, schemes.hybrid_rush_larsen, stiff_states=["V"]
    )
    assert inlined == via_intermediate
    assert "dV_dt_linearized = -g" in inlined
