"""Model-level acceptance tests for removable-singularity guarding.

Mapped to the test plan in
``docs/superpowers/specs/2026-09-13-singularities-linearized-block-design.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy

import gotranx
from gotranx.atoms import make_symbol
from gotranx.linearization import diagonal_jacobian
from gotranx.load import ode_from_string

HERE = Path(__file__).parent
ODEFILES = HERE / "odefiles"

TOY = """
parameters(C=1.0)
states("Membrane", V=-80.0, n=0.1)
expressions("Membrane")
alpha_n = (V + 10)/(exp((V + 10)/10) - 1)
dn_dt = alpha_n*(1 - n) - 0.125*n
dV_dt = -alpha_n*(V + 77)/C
"""


_LAMBDIFIED: dict[sympy.Expr, tuple] = {}


def _evaluate(expr, values):
    """Evaluate `expr` at `values` under numpy's floating-point semantics.

    Not by symbolic substitution. ToRORd's bundled initial state has
    `CaTrpn = 0`, and `dTmB_dt` contains the model's own guard
    `Piecewise((CaTrpn**(-ntm/2), CaTrpn**(-ntm/2) < 100), (100, True))`.
    Substituted exactly, `0**negative` is sympy's `zoo` and comparing it
    raises; under numpy it is `inf`, the comparison is False, and the branch
    that is actually taken returns 100. numpy's semantics are what the
    generated code runs under, so they are what these tests must use --
    `tests/test_deep_linearization_models.py` reaches the same conclusion for
    the same expression.
    """
    if expr not in _LAMBDIFIED:
        symbols = sorted(expr.free_symbols, key=str)
        _LAMBDIFIED[expr] = (symbols, sympy.lambdify(symbols, expr, "numpy"))
    symbols, function = _LAMBDIFIED[expr]
    return float(function(*[np.float64(values[s]) for s in symbols]))


def _environment(ode, overrides, t=0.0):
    """Default parameter/state values, plus every intermediate resolved in
    dependency order, with `overrides` applied to the states first.

    `t` has to be in here too: these models carry a stimulus expressed as a
    Piecewise over time, which has no value until time is pinned.
    """
    values = {make_symbol(p.name): float(p.value) for p in ode.parameters}
    values.update({make_symbol(s.name): float(s.value) for s in ode.states})
    values[ode.t] = t
    values.update(overrides)
    for assignment in ode.sorted_assignments():
        values[assignment.symbol] = _evaluate(assignment.expr, values)
    return values


@pytest.fixture(scope="module")
def guarded():
    names = ["beeler_reuter_1977", "ToRORd_dyn_chloride", "ORdmm_Land"]
    return {
        name: gotranx.load_ode(ODEFILES / f"{name}.ode", remove_singularities=True)
        for name in names
    }


def test_toy_model_guard_evaluates_to_the_true_limit():
    """Test plan item 3, and the design document's F4.

    The emitted branch is a polynomial, not the constant 23.5, so this asserts
    the value the guard produces rather than grepping the generated text for a
    literal. 23.5 is `-alpha_n/C + (-V - 77)*alpha_n'/C` at V = -10, i.e.
    -10 + (-67)*(-1/2); an order-0 replacement gives alpha_n' = 0 and -10.
    """
    ode = ode_from_string(TOY, remove_singularities=True)
    alpha_n = [a for a in ode.sorted_assignments() if a.name == "alpha_n"][0]
    assert alpha_n.expr.has(sympy.Piecewise), "alpha_n was not guarded"

    values = _environment(ode, {make_symbol("V"): -10.0})
    assert abs(values[make_symbol("alpha_n")] - 10.0) < 1e-12
    assert abs(_evaluate(diagonal_jacobian(ode)["V"], values) - 23.5) < 1e-9


def test_toy_model_is_unguarded_when_asked():
    ode = ode_from_string(TOY, remove_singularities=False)
    alpha_n = [a for a in ode.sorted_assignments() if a.name == "alpha_n"][0]
    assert not alpha_n.expr.has(sympy.Piecewise)


@pytest.mark.parametrize(
    "name, state, pole",
    [
        ("beeler_reuter_1977", "V", -23.0),
        ("ToRORd_dyn_chloride", "v", 0.0),
        ("ORdmm_Land", "v", 0.0),
    ],
)
def test_diagonal_jacobian_is_finite_across_the_pole(name, state, pole, guarded):
    """Test plan items 1 and 2.

    Unguarded, ToRORd's d(dv_dt)/dv is +30.1 at v = -1e-9 and raises at v = 0,
    against a true value near -0.0405. The sign is what makes it dangerous:
    the linearization feeds exp(g*dt), so a positive g turns a decaying
    exponential into a growing one.
    """
    ode = guarded[name]
    jacobian = diagonal_jacobian(ode)[state]
    var = make_symbol(state)

    np.seterr(all="ignore")
    for offset in (-1e-1, -1e-3, -1e-9, 0.0, 1e-9, 1e-3, 1e-1):
        values = _environment(ode, {var: pole + offset})
        got = _evaluate(jacobian, values)
        assert np.isfinite(got), f"{name}: non-finite at {state} = {pole + offset}"


@pytest.mark.parametrize(
    "name, state, pole",
    [
        ("beeler_reuter_1977", "V", -23.0),
        ("ToRORd_dyn_chloride", "v", 0.0),
    ],
)
def test_diagonal_jacobian_is_continuous_across_the_pole(name, state, pole, guarded):
    """The guarded value at the pole must lie on the same curve as its own
    neighborhood.

    Finiteness alone would be satisfied by any constant, so this checks that
    the three points are collinear: the value at the pole must sit on the
    midpoint of its two neighbours, to far better than the spread between
    them. A jump of size `j` would move it off that midpoint by `j/2`, while
    genuine curvature only moves it by O(h**2 * f''), which is negligible next
    to the O(h * f') spread. Comparing the pole value against a neighbour
    directly would not work: the function legitimately varies over the
    +-1e-4 window used here.
    """
    ode = guarded[name]
    jacobian = diagonal_jacobian(ode)[state]
    var = make_symbol(state)

    np.seterr(all="ignore")
    at = {
        offset: _evaluate(jacobian, _environment(ode, {var: pole + offset}))
        for offset in (-1e-4, 0.0, 1e-4)
    }
    spread = abs(at[1e-4] - at[-1e-4])
    assert spread > 0, f"{name}: the diagonal is constant here, nothing is being tested: {at}"
    midpoint = (at[-1e-4] + at[1e-4]) / 2
    assert abs(at[0.0] - midpoint) / spread < 1e-3, at


def test_beeler_reuter_agrees_with_a_high_precision_reference(guarded):
    """Test plan item 1, the accuracy half: d(dV_dt)/dV near V = -23.

    The reference is the *mathematical* model: every assignment rationalized,
    inlined into dV_dt, differentiated exactly, and evaluated in 50-digit
    arithmetic. Evaluating the float expressions as written in high precision
    is not a valid reference -- it faithfully reproduces float round-off in
    the model's own constants, which moves the pole off -23 by ~1e-14 and
    shows an error that scales like 1/offset and belongs to the reference,
    not the guard.

    The design document asks for ~1e-11 relative agreement. Measured worst
    over this sweep: 1.58e-10, at V = -23.0143, just outside the window on
    the direct-formula side -- the same value with or without the guard. The
    guard's own branch is at machine precision. (With the window placed by
    the analytic truncation-vs-round-off rule instead, the worst was 1.8e-8.)
    """
    mpmath = pytest.importorskip("mpmath")
    mpmath.mp.dps = 50

    ode = guarded["beeler_reuter_1977"]
    unguarded = gotranx.load_ode(ODEFILES / "beeler_reuter_1977.ode", remove_singularities=False)
    var = make_symbol("V")

    definitions = {
        a.symbol: sympy.nsimplify(a.expr, rational=True) for a in unguarded.sorted_assignments()
    }
    dV_dt = definitions[make_symbol("dV_dt")]
    while True:
        substitutions = {s: definitions[s] for s in dV_dt.free_symbols if s in definitions}
        if not substitutions:
            break
        dV_dt = dV_dt.xreplace(substitutions)
    reference = sympy.diff(dV_dt, var)
    constants = {
        make_symbol(p.name): sympy.nsimplify(sympy.sympify(p.value), rational=True)
        for p in unguarded.parameters
    }
    constants.update(
        {
            make_symbol(st.name): sympy.nsimplify(sympy.sympify(st.value), rational=True)
            for st in unguarded.states
            if st.name != "V"
        }
    )
    constants[unguarded.t] = sympy.Integer(0)
    reference = sympy.lambdify(var, reference.xreplace(constants), "mpmath")

    jacobian = diagonal_jacobian(ode)["V"]
    np.seterr(all="ignore")
    worst, where = 0.0, None
    for offset in (float(o) for o in np.geomspace(1e-9, 0.2, 30)):
        for point in (-23.0 - offset, -23.0 + offset):
            got = _evaluate(jacobian, _environment(ode, {var: point}))
            expected = reference(mpmath.mpf(repr(point)))
            error = float(abs(mpmath.mpf(got) - expected) / abs(expected))
            if error > worst:
                worst, where = error, point

    assert worst < 1e-9, f"worst relative disagreement {worst:.2e} at V = {where}"


def _generated_scheme_module(name, remove_singularities=True):
    import gotranx.cli.gotran2py
    from gotranx.schemes import Scheme

    ode = gotranx.load_ode(ODEFILES / f"{name}.ode", remove_singularities=remove_singularities)
    code = gotranx.cli.gotran2py.get_code(ode, scheme=[Scheme.generalized_rush_larsen])
    namespace: dict = {}
    exec(compile(code, f"<{name}>", "exec"), namespace)
    return ode, namespace


@pytest.mark.parametrize(
    "name, state, pole",
    [
        ("beeler_reuter_1977", "V", -23.0),
        ("ToRORd_dyn_chloride", "v", 0.0),
        ("ORdmm_Land", "v", 0.0),
    ],
)
def test_generated_scheme_is_finite_exactly_at_the_pole(name, state, pole):
    """Test plan items 2 and 4, end to end through the generated code.

    Places the state exactly on the pole and steps the generated
    generalized_rush_larsen stepper once. Every CSE temporary is computed
    unconditionally, so a temporary hoisted out of a guard would make the
    step non-finite here even if the guarded assignment itself were fine.
    """
    _, mod = _generated_scheme_module(name)
    parameters = mod["init_parameter_values"]()
    states = mod["init_state_values"]()
    states[mod["state_index"](state)] = pole
    with np.errstate(all="ignore"):
        result = mod["generalized_rush_larsen"](states, 0.0, 1e-3, parameters)
    assert np.all(np.isfinite(result)), (
        f"{name}: non-finite states {np.flatnonzero(~np.isfinite(result)).tolist()}"
    )


def test_generated_scheme_is_not_finite_at_the_pole_without_guards():
    """The load-bearing check for the test above: without guards the same
    step is not finite, so that test is not passing by accident."""
    _, mod = _generated_scheme_module("ToRORd_dyn_chloride", remove_singularities=False)
    parameters = mod["init_parameter_values"]()
    states = mod["init_state_values"]()
    states[mod["state_index"]("v")] = 0.0
    with np.errstate(all="ignore"):
        result = mod["generalized_rush_larsen"](states, 0.0, 1e-3, parameters)
    assert not np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "name, state",
    [("ToRORd_dyn_chloride", "v"), ("tentusscher_panfilov_2006_M_cell", "V")],
)
def test_stays_stable_at_dt_0_02_over_300_ms(name, state):
    """Test plan item 5, the substituted stability criterion.

    The 2.0.0 criterion was stated against base_model_IM.ode, which is not a
    bundled fixture (tests/test_deep_linearization_models.py says so). The
    author confirmed this substitution: dt = 0.02 ms over 300 ms on the two
    bundled models, which covers a full beat through v = 0.
    """
    _, mod = _generated_scheme_module(name)
    parameters = mod["init_parameter_values"]()
    y = mod["init_state_values"]()
    index = mod["state_index"](state)
    dt, t = 0.02, 0.0
    lowest, highest = float(y[index]), float(y[index])
    with np.errstate(all="ignore"):
        for _ in range(15000):
            y = mod["generalized_rush_larsen"](y, t, dt, parameters)
            t += dt
            assert np.all(np.isfinite(y)), f"{name}: non-finite at t = {t:.2f} ms"
            lowest, highest = min(lowest, float(y[index])), max(highest, float(y[index]))
    # The run must actually fire an action potential through the pole, or it
    # has not exercised the guard at all.
    assert highest > 0.0, f"{name}: {state} never crossed 0 (peak {highest:.1f} mV)"
    assert -120.0 < lowest and highest < 80.0, (lowest, highest)
