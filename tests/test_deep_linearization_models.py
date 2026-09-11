"""Model-level checks that linearization is deep, correct, and cheap.

These complement the string assertions in test_schemes.py, which use
three-state fixtures and cannot show that a real cardiac model integrates
correctly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy

import gotranx
from gotranx.codegen import PythonCodeGenerator
from gotranx.linearization import diagonal_jacobian
from gotranx.schemes import get_scheme

HERE = Path(__file__).parent
ODEFILES = HERE / "odefiles"

# Measured expectations. The deep column is the point of this change; the
# shallow column is what gotranx produced before it.
EXPECTED_NONZERO = {
    "lorentz": (3, 3),
    "fitzhughnagumo": (1, 2),
    "beeler_reuter_1977": (7, 8),
    "tentusscher_panfilov_2006_M_cell": (13, 19),
    "ORdmm_Land": (39, 48),
    "ToRORd_dyn_chloride": (41, 52),
}

# Captured from the pre-change (Task 0, before Task 1) code generator. lorentz.ode
# writes its derivatives directly rather than through intermediates, so deep
# linearization must be a byte-for-byte no-op for it. Do not regenerate this
# constant from the current codebase -- that would defeat the point of the
# canary, since output produced after the change cannot prove the change was
# a no-op.
EXPECTED_LORENTZ_SCHEME = """def generalized_rush_larsen(states, t, dt, parameters):

    # Assign states
    x = states[0]
    y = states[1]
    z = states[2]

    # Assign parameters
    beta = parameters[0]
    rho = parameters[1]
    sigma = parameters[2]

    # Assign expressions

    values = numpy.zeros_like(states, dtype=numpy.float64)
    dx_dt = sigma * (-x + y)
    dx_dt_linearized = -sigma
    values[0] = x + numpy.where(
        numpy.logical_or((dx_dt_linearized > 1e-08), (dx_dt_linearized < -1e-08)),
        dx_dt * (numpy.exp(dt * dx_dt_linearized) - 1) / dx_dt_linearized,
        dt * dx_dt,
    )
    dy_dt = x * (rho - z) - y
    dy_dt_linearized = -1
    values[1] = y + numpy.where(
        numpy.logical_or((dy_dt_linearized > 1e-08), (dy_dt_linearized < -1e-08)),
        dy_dt * (numpy.exp(dt * dy_dt_linearized) - 1) / dy_dt_linearized,
        dt * dy_dt,
    )
    dz_dt = -beta * z + x * y
    dz_dt_linearized = -beta
    values[2] = z + numpy.where(
        numpy.logical_or((dz_dt_linearized > 1e-08), (dz_dt_linearized < -1e-08)),
        dz_dt * (numpy.exp(dt * dz_dt_linearized) - 1) / dz_dt_linearized,
        dt * dz_dt,
    )

    return values
"""


@pytest.fixture(scope="module")
def loaded():
    return {name: gotranx.load_ode(ODEFILES / f"{name}.ode") for name in EXPECTED_NONZERO}


@pytest.mark.parametrize("name", sorted(EXPECTED_NONZERO))
def test_every_state_of_every_bundled_model_is_linearized(name, loaded):
    """No state may silently fall back to forward Euler."""
    ode = loaded[name]
    _, expected_deep = EXPECTED_NONZERO[name]
    jac = diagonal_jacobian(ode)
    nonzero = sorted(k for k, v in jac.items() if not v.is_zero)
    assert len(jac) == len(ode.states)
    assert len(nonzero) == expected_deep, f"zero diagonal for {sorted(set(jac) - set(nonzero))}"


@pytest.mark.parametrize("name", sorted(EXPECTED_NONZERO))
def test_shallow_linearization_was_incomplete(name, loaded):
    """Pins what the old behavior was, so the size of the fix stays visible."""
    ode = loaded[name]
    expected_shallow, _ = EXPECTED_NONZERO[name]
    shallow = [a for a in ode.state_derivatives if not sympy.diff(a.expr, a.state.symbol).is_zero]
    assert len(shallow) == expected_shallow


def test_lorentz_scheme_is_unchanged(loaded):
    """lorentz.ode writes its derivatives directly, so deep linearization must
    be a no-op for it. This is the canary that the change is surgical."""
    generated = PythonCodeGenerator(loaded["lorentz"]).scheme(get_scheme("generalized_rush_larsen"))
    assert generated == EXPECTED_LORENTZ_SCHEME


def _generated_module(ode, name):
    import gotranx.cli.gotran2py

    code = gotranx.cli.gotran2py.get_code(
        ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen]
    )
    namespace: dict = {}
    exec(compile(code, f"<{name}>", "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", ["tentusscher_panfilov_2006_M_cell", "ToRORd_dyn_chloride"])
def test_derivatives_match_central_differences_through_rhs(name, loaded):
    """The symbolic diagonal must equal the numerical diagonal of the generated
    rhs, at several points along a trajectory."""
    ode = loaded[name]
    mod = _generated_module(ode, name)
    jac = diagonal_jacobian(ode)

    np.seterr(all="ignore")
    parameters = mod["init_parameter_values"]()
    base = mod["init_state_values"]()
    rng = np.random.default_rng(0)
    # Both a relative and an absolute component: several states (and monitored
    # quantities that depend only on parameters, e.g. ToRORd's `C`) sit at exactly
    # 0 in the bundled initial conditions, so a purely relative perturbation
    # never moves them and every sample point lands exactly on a Conditional's
    # branch boundary (e.g. ToRORd's `eta = Conditional(Lt(C - Cd, 0), ...)`,
    # where C and Cd are both pinned to 0). A central difference straddling such
    # a kink disagrees with the one-sided analytic derivative by construction,
    # which is a property of the sampling point, not of the AD sweep. The
    # absolute term breaks that tie without materially changing the trajectory.
    points = [
        (
            base * (1 + 0.03 * rng.standard_normal(len(base)))
            + 1e-3 * rng.standard_normal(len(base)),
            5.0 * k,
        )
        for k in (0, 1, 2)
    ]

    worst = 0.0
    comparisons = 0
    for state in ode.states:
        expr = jac[state.name]
        for values, t in points:
            monitor = mod["monitor_values"](t, values, parameters)
            # Keep numpy scalar dtypes (not Python floats) all the way through
            # lambdify: several monitored expressions raise a state to a
            # negative power that is only ever reached inside a numpy.select
            # branch that is not actually taken at runtime (e.g. ToRORd's
            # kb*select(...) guard). numpy evaluates both branches and relies on
            # 0.0 ** negative -> inf under its own floating-point semantics
            # (suppressed by seterr above); plain Python floats raise
            # ZeroDivisionError for the same expression instead.
            env = {s.symbol: np.float64(values[mod["state_index"](s.name)]) for s in ode.states}
            env.update(
                {
                    p.symbol: np.float64(parameters[mod["parameter_index"](p.name)])
                    for p in ode.parameters
                }
            )
            for a in ode.sorted_assignments():
                try:
                    env[a.symbol] = np.float64(monitor[mod["monitor_index"](a.name)])
                except KeyError:
                    pass
            env[sympy.Symbol("time", real=True)] = np.float64(t)
            if any(s not in env for s in expr.free_symbols):
                continue

            symbolic = float(sympy.lambdify(list(env), expr, "numpy")(*env.values()))
            i = mod["state_index"](state.name)
            h = 1e-6 * max(1.0, abs(values[i]))
            up, down = values.copy(), values.copy()
            up[i] += h
            down[i] -= h
            numeric = (mod["rhs"](t, up, parameters)[i] - mod["rhs"](t, down, parameters)[i]) / (
                2 * h
            )
            worst = max(worst, abs(symbolic - numeric) / max(1e-6, abs(numeric)))
            comparisons += 1

    assert comparisons >= len(ode.states), "too few states resolved to be meaningful"
    assert worst < 1e-4, f"worst relative disagreement {worst:.2e} over {comparisons} comparisons"


@pytest.mark.parametrize("name", ["tentusscher_panfilov_2006_M_cell", "fitzhughnagumo"])
def test_models_using_conditionals_generate_and_differentiate(name, loaded):
    """Conditional -> Piecewise differentiates branch-wise; ContinuousConditional
    differentiates to a smooth function. Neither needs special casing, but both
    must survive the AD sweep and reach the generated code."""
    ode = loaded[name]
    jac = diagonal_jacobian(ode)
    for expr in jac.values():
        assert not expr.has(sympy.Derivative)
        assert not expr.has(sympy.DiracDelta)
    generated = PythonCodeGenerator(ode).scheme(get_scheme("generalized_rush_larsen"))
    assert "_linearized" in generated


def test_linearization_block_stays_within_twice_the_rhs(loaded):
    """The linearization block -- the per-state-CSE'd Jacobian diagonal that
    backs the `d<state>_dt_linearized` assignments -- must stay within 2x the
    plain rhs, measured by `sympy.count_ops` the same way the design's
    strategy table measured it (raw, un-CSE'd derivatives were 9.4x rhs).

    A whole-*scheme* 2x bound is unachievable for ToRORd, structurally: a
    scheme necessarily contains the entire rhs plus this block, so
    scheme/rhs = 1 + block/rhs. Getting the whole function under 2x needs the
    block itself at <= 1.0x the rhs; per-state CSE (chosen over joint CSE on
    liveness grounds, see `_linearized_assignments`) gets to 1.35x on ToRORd,
    and even joint CSE only reaches 1.08x. So the block, not the whole
    function, is the thing this test bounds. The whole-scheme assertion below
    is a deliberately loose line-count ceiling kept only as an explosion
    guard for a future change in emission strategy -- not a quality bar.

    Measured at the time of writing:
        ToRORd_dyn_chloride:              block 1.35x; whole 2.35x by ops, 3.13x by lines
        tentusscher_panfilov_2006_M_cell: block 0.88x; whole 1.88x by ops
        base_model_IM (design doc, not a bundled fixture): block 0.69x; whole 1.69x
            (matches legacy gotran's 1.7x measured on the same model)
    """
    ode = loaded["ToRORd_dyn_chloride"]
    jac = diagonal_jacobian(ode)

    rhs_ops = sum(sympy.count_ops(a.expr) for a in ode.sorted_assignments())

    block_ops = 0
    for state in ode.states:
        replacements, reduced = sympy.cse([jac[state.name]], optimizations="basic")
        for _, sub_expr in replacements:
            block_ops += sympy.count_ops(sub_expr)
        block_ops += sympy.count_ops(reduced[0])

    assert block_ops < 2 * rhs_ops, f"{block_ops} vs {rhs_ops} rhs ops"

    codegen = PythonCodeGenerator(ode)
    rhs_lines = len([ln for ln in codegen.rhs().splitlines() if ln.strip()])
    scheme_lines = len(
        [
            ln
            for ln in codegen.scheme(get_scheme("generalized_rush_larsen")).splitlines()
            if ln.strip()
        ]
    )
    assert scheme_lines < 4 * rhs_lines, f"{scheme_lines} vs {rhs_lines} rhs lines"
