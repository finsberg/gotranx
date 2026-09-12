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


def test_generalized_rush_larsen_emits_repeated_subexpression_as_temporary(parser, trans):
    """A Jacobian diagonal entry with a genuine repeated subexpression must be
    factored into a temporary, even for a lone state with nothing to share with.

    ``d/dx[sin(x**2)*cos(x**2)]`` is ``2*x*cos(x**2)**2 - 2*x*sin(x**2)**2``,
    in which ``x**2`` occurs four times; confirmed with a standalone
    ``sympy.cse`` call before writing this test. Every Jacobian entry reached
    by the other scheme tests is a single symbol, so none of them exercise the
    temporary-emission loop or the `fresh()` name generator -- this test is
    the one that does, and it is the single-state counterpart to
    `test_generalized_rush_larsen_cse_shares_temporary_across_states`, which
    covers factoring *between* states.
    """
    expr = """
    states(x=0.5)
    dx_dt = sin(x**2)*cos(x**2)
    """
    ode = make_ode(*trans.transform(parser.parse(expr)))
    dt = sympy.Symbol("dt")
    eqs = [str(e) for e in schemes.generalized_rush_larsen(ode, dt)]

    temp_indices = [i for i, e in enumerate(eqs) if e.startswith("_linearization_temp_")]
    assert temp_indices, f"expected at least one CSE temporary, got none: eqs={eqs}"

    state_derivative_index = next(i for i, e in enumerate(eqs) if e.startswith("dx_dt = "))
    linearized_index = next(i for i, e in enumerate(eqs) if e.startswith("dx_dt_linearized = "))

    # A temporary is only safe to emit inline if it comes after the state
    # derivative it was factored out of, and before the `_linearized` line
    # that consumes it.
    assert state_derivative_index < min(temp_indices)
    assert max(temp_indices) < linearized_index

    temp_names = [eqs[i].split(" = ")[0] for i in temp_indices]
    assert any(temp_name in eqs[linearized_index] for temp_name in temp_names)


def test_linearized_assignments_skips_taken_names():
    """`_linearized_assignments`'s `fresh()` generator (`_linearization_temp_{i}`)
    must skip names already in `taken`, not merely start counting past them.

    `sympy.cse(..., symbols=...)` consumes the generator lazily, one name per
    membership check, so "start past the taken count" and "skip taken names"
    only coincide when `taken` is contiguous from `_0`.
    """
    x = sympy.Symbol("x")
    expr = -2 * x * sympy.sin(x**2) ** 2 + 2 * x * sympy.cos(x**2) ** 2

    plan = schemes._linearized_assignments([("dx_dt", expr)], taken={"_linearization_temp_0"})
    replacements, _ = plan["dx_dt"]

    assert replacements
    assert replacements[0][0].name == "_linearization_temp_1"


def test_hybrid_rush_larsen_restricts_cse_to_stiff_linearizable_states(parser, trans, monkeypatch):
    """Nothing else pins that `hybrid_rush_larsen` feeds only stiff,
    non-zero-diagonal states into the CSE plan -- a state that falls
    back to forward Euler must contribute no temporary and no linearized
    line, even when its own (unused) derivative happens to share the exact
    subexpression the stiff states' diagonal entries share.

    Three states: `x` and `y` are stiff and their diagonal Jacobian entries
    both reduce to the same expression (as in
    `test_generalized_rush_larsen_cse_shares_temporary_across_states`),
    so CSE must factor one shared temporary for them. `z` is *not*
    stiff, but its derivative is written with the identical subexpression
    `sin(x + y)*cos(x + y)` -- the shape the reviewer used to try to break the
    stiff-subset restriction. If a future change fed all linearizable states
    (not just the stiff ones) into the plan, `z` could end up claiming
    the shared temporary's emission slot instead of `x`/`y` -- `z`'s branch
    never emits the plan's replacements (it takes the forward-Euler path), so
    the temporary `x`/`y` need would silently vanish. A `monkeypatch` spy on
    `_linearization_plan` pins the actual restriction directly, rather than
    relying on that failure mode reproducing reliably through output alone.
    """
    expr = """
    states(x=1.0, y=1.0, z=1.0)
    dx_dt = sin(x + y)*cos(x + y) - x
    dy_dt = sin(x + y)*cos(x + y) - y
    dz_dt = sin(x + y)*cos(x + y) - z
    """
    ode = make_ode(*trans.transform(parser.parse(expr)))
    dt = sympy.Symbol("dt")

    seen_items = []
    real_linearization_plan = schemes._linearization_plan

    def spy(items, taken, cse):
        seen_items.append([name for name, _ in items])
        return real_linearization_plan(items, taken, cse)

    monkeypatch.setattr(schemes, "_linearization_plan", spy)

    eqs = [str(e) for e in schemes.hybrid_rush_larsen(ode, dt, stiff_states=["x", "y"])]

    assert seen_items == [["dx_dt", "dy_dt"]], (
        f"expected only the stiff, linearizable states fed into the plan: {seen_items}"
    )

    assert not [e for e in eqs if e.startswith("dz_dt_linearized")]
    assert "values[2] = dt*dz_dt + z" in eqs

    temp_lines = [e for e in eqs if e.startswith("_linearization_temp_")]
    assert temp_lines, f"expected a CSE temporary shared by x and y: eqs={eqs}"

    dx_linearized = next(e for e in eqs if e.startswith("dx_dt_linearized = "))
    dy_linearized = next(e for e in eqs if e.startswith("dy_dt_linearized = "))
    temp_names = [e.split(" = ")[0] for e in temp_lines]
    assert any(name in dx_linearized for name in temp_names)
    assert any(name in dy_linearized for name in temp_names)


def test_generalized_rush_larsen_cse_is_on_by_default():
    """CSE costs fewer operations than not doing it on every model measured,
    so it should not need to be opted into. `cse=False` exists for the one
    thing CSE costs: a named local per factored subexpression, which on the
    vectorized numpy backend is a live array."""
    import inspect

    sig = inspect.signature(schemes.generalized_rush_larsen)
    assert sig.parameters["cse"].default is True


def test_hybrid_rush_larsen_cse_is_on_by_default():
    import inspect

    sig = inspect.signature(schemes.hybrid_rush_larsen)
    assert sig.parameters["cse"].default is True


def test_cse_false_inlines_the_full_expression_with_no_temporaries(parser, trans):
    """`cse=False` must not factor anything: `d<state>_dt_linearized` is one
    inlined expression, with no temporaries at all."""
    expr = """
    states(x=0.5)
    dx_dt = sin(x**2)*cos(x**2)
    """
    ode = make_ode(*trans.transform(parser.parse(expr)))
    dt = sympy.Symbol("dt")
    eqs = [str(e) for e in schemes.generalized_rush_larsen(ode, dt, cse=False)]

    assert not [e for e in eqs if e.startswith("_linearization_temp_")]
    linearized = next(e for e in eqs if e.startswith("dx_dt_linearized = "))
    # The raw, un-factored diagonal Jacobian entry.
    assert linearized == "dx_dt_linearized = -2*x*math.sin(x**2)**2 + 2*x*math.cos(x**2)**2"


SHARED_SUBEXPRESSION = """
states(x=1.0, y=1.0)
q = sin(x + y)*cos(x + y)
dx_dt = q - x
dy_dt = q - y
"""


def test_generalized_rush_larsen_cse_shares_temporary_across_states(parser, trans):
    """The crux of CSE here: a subexpression shared *between* two states'
    linearized derivatives must be computed once, as a `_linearization_temp_*`
    temporary emitted immediately before the first state that needs it (here,
    x), and simply reused (not recomputed) for the second (y).

    `d(dx_dt)/dx` and `d(dy_dt)/dy` are both `cos(x+y)**2 - sin(x+y)**2 - 1`
    here (since `d(x+y)/dx == d(x+y)/dy == 1`), confirmed with a standalone
    `sympy.cse` call before writing this test: `sympy.cse([jac_x, jac_y])`
    returns one temporary for `x+y` and a second temporary, identical for
    both states, for the whole reduced expression.
    """
    ode = make_ode(*trans.transform(parser.parse(SHARED_SUBEXPRESSION)))
    dt = sympy.Symbol("dt")
    eqs = [str(e) for e in schemes.generalized_rush_larsen(ode, dt)]  # default: cse on

    temp_indices = [i for i, e in enumerate(eqs) if e.startswith("_linearization_temp_")]
    assert temp_indices, f"expected CSE temporaries, got none: eqs={eqs}"

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
    ode = make_ode(*trans.transform(parser.parse(SHARED_SUBEXPRESSION)))
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
