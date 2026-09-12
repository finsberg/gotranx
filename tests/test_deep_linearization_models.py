"""Model-level checks that linearization is deep, correct, and cheap.

These complement the string assertions in test_schemes.py, which use
three-state fixtures and cannot show that a real cardiac model integrates
correctly.
"""

from __future__ import annotations

import sys
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


def _generated_module(ode, name, schemes=(gotranx.schemes.Scheme.generalized_rush_larsen,)):
    import gotranx.cli.gotran2py

    code = gotranx.cli.gotran2py.get_code(ode, scheme=list(schemes))
    namespace: dict = {}
    exec(compile(code, f"<{name}>", "exec"), namespace)
    return namespace


def _generated_module_with_cse(ode, name, cse):
    """Like `_generated_module`, but reaches `PythonCodeGenerator.scheme`
    directly so a `cse` kwarg can be threaded through to
    `generalized_rush_larsen` without going through the CLI layer that
    `_generated_module` uses.
    """
    from gotranx.codegen.python import Format

    codegen = PythonCodeGenerator(ode, format=Format.none)
    comp = [
        codegen.imports(),
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.missing_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
        codegen.scheme(get_scheme("generalized_rush_larsen"), cse=cse),
    ]
    code = codegen._format("\n".join(comp))
    namespace: dict = {}
    exec(compile(code, f"<{name}-{cse}>", "exec"), namespace)
    return namespace


def _call_capturing_locals(func, *args):
    """Call a generated scheme function and return `(result, locals_at_return)`.

    A generated scheme function computes many local temporaries -- among them
    the post-CSE `d<state>_dt_linearized` that backs each Rush-Larsen update --
    but returns only `values`. `sys.settrace` with a return-hook on the
    function's own code object is the only way to see those temporaries
    without changing codegen: it snapshots `frame.f_locals` right before the
    function returns, without altering what the function computes or returns.
    """
    target_code = func.__code__
    captured: dict = {}

    def tracer(frame, event, arg):
        if event == "call" and frame.f_code is target_code:

            def local_tracer(frame, event, arg):
                if event == "return":
                    captured.update(frame.f_locals)
                return local_tracer

            return local_tracer
        return None

    old_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = func(*args)
    finally:
        sys.settrace(old_trace)
    return result, captured


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

    # Measured: every one of the 3 sample points resolves for every state, with
    # zero skips (the `any(s not in env for s in expr.free_symbols)` guard
    # never trips for these two models). Tightened from a `>=` floor -- which
    # could pass with only 1 of 3 points resolving per state -- to the exact
    # count, so the assertion says what it means.
    assert comparisons == 3 * len(ode.states), (
        f"expected {3 * len(ode.states)} comparisons with zero skips, got {comparisons}"
    )
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
    block itself at <= 1.0x the rhs, and CSE only reaches 1.08x on ToRORd. So
    the block, not the whole function, is the thing this test bounds. The
    whole-scheme assertion below is a deliberately loose line-count ceiling
    kept only as an explosion guard for a future change in emission strategy
    -- not a quality bar.

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


@pytest.mark.parametrize("name", ["ToRORd_dyn_chloride", "tentusscher_panfilov_2006_M_cell"])
def test_generated_scheme_executes_and_agrees_with_explicit_euler(name, loaded):
    """Actually *run* the generated `generalized_rush_larsen` stepper.

    Every other test in this module either inspects `diagonal_jacobian(ode)`
    directly (pre-CSE, never generated) or `exec`s the generated module without
    calling anything in it. `exec` only compiles the scheme function -- it
    does not execute its body, so a use-before-definition among the CSE
    temporaries (`fresh()` colliding with a real symbol, or a temporary
    emitted after the expression that uses it) would raise `NameError` only
    when the function is actually *called*, and no existing test calls it.
    This test closes that gap: it steps the generated stepper for real, from
    real initial conditions, thousands of times, and checks that it runs to
    completion and stays finite.

    What this test does NOT show: that the emitted `d<state>_dt_linearized`
    has the *correct value*. GRL and explicit Euler differ by O(g * dt**2) per
    step, and at the dt = 1e-3 ms used here -- small enough to keep explicit
    Euler itself stable, which both models require to make the comparison
    below meaningful at all -- that error term is negligible regardless of
    whether `g` is right, wrong, or sign-flipped. Measured: negating
    `diagonal_jacobian` entirely, or scaling it by 2x, both still pass the
    `atol=rtol=1e-2` check below (worst-case observed drift unchanged to 3
    significant figures). So this test's agreement assertion is a "did the
    scheme run and stay in a physiologically sane basin" smoke check, not a
    correctness check on the linearization -- that numeric burden is carried
    by `test_emitted_linearized_expression_matches_diagonal_jacobian` below,
    which was demonstrated (see the fix-report) to fail hard when `g` is
    perturbed the same way.

    dt = 1e-3 ms / 2000 steps was chosen because it is small enough that both
    schemes are comfortably inside explicit Euler's stability region for
    these two models (explicit Euler already goes unstable by dt =
    3e-3..5e-3 on both) -- a prerequisite for this smoke check, not evidence
    of numerical agreement between the two update rules' linearization terms.
    """
    ode = loaded[name]
    mod = _generated_module(
        ode,
        name,
        schemes=(
            gotranx.schemes.Scheme.generalized_rush_larsen,
            gotranx.schemes.Scheme.explicit_euler,
        ),
    )

    np.seterr(all="ignore")
    parameters = mod["init_parameter_values"]()
    y_grl = mod["init_state_values"]()
    y_euler = y_grl.copy()

    dt = 1e-3
    n_steps = 2000
    t = 0.0
    for _ in range(n_steps):
        y_grl = mod["generalized_rush_larsen"](y_grl, t, dt, parameters)
        y_euler = mod["explicit_euler"](y_euler, t, dt, parameters)
        assert np.all(np.isfinite(y_grl)), f"non-finite state at t={t}"
        assert np.all(np.isfinite(y_euler)), f"non-finite explicit_euler state at t={t}"
        t += dt

    np.testing.assert_allclose(
        y_grl,
        y_euler,
        atol=1e-2,
        rtol=1e-2,
        err_msg=(
            f"generalized_rush_larsen disagrees with explicit_euler for {name} "
            f"at dt={dt} after {n_steps} steps"
        ),
    )


@pytest.mark.parametrize("name", ["tentusscher_panfilov_2006_M_cell", "ToRORd_dyn_chloride"])
def test_emitted_linearized_expression_matches_diagonal_jacobian(name, loaded):
    """The post-CSE `d<state>_dt_linearized` value the generated code actually
    computes must equal `diagonal_jacobian`'s pre-CSE symbolic value, at the
    same point.

    This is the numeric link nothing else in this module checks.
    `test_generated_scheme_executes_and_agrees_with_explicit_euler` proves the
    generated code runs to completion without a `NameError`, but its agreement
    check against `explicit_euler` is *not* sensitive to the value of the
    linearization -- GRL and forward Euler differ by O(g * dt**2) per step,
    negligible at the dt small enough to keep forward Euler itself stable.
    Verified by hand: negating `diagonal_jacobian` everywhere, or doubling it,
    both still pass that test's `atol=rtol=1e-2` check (see the fix-report for
    the actual before/after numbers). This test closes that gap directly by
    comparing the value CSE and emission actually produced against the
    untouched symbolic diagonal, so it is sensitive to exactly the class of
    bug the other test cannot see.

    Captures the emitted `d<state>_dt_linearized` locals via
    `_call_capturing_locals` (the function returns only `values`, not its
    temporaries) and compares each one against `sympy.lambdify` of
    `diagonal_jacobian(ode)[state]` evaluated at the same
    state/parameter/monitor values. Uses the bundled `init_state_values()` at
    three different `t` (the models are time-dependent only through an
    externally supplied stimulus protocol, so varying `t` alone still
    exercises different code paths) rather than randomly perturbed states:
    perturbing states by the +-3% used elsewhere in this module pushes some
    of ToRORd's concentration-like states into a domain where an unrelated
    monitored quantity (not the linearization) legitimately evaluates to NaN
    on both the symbolic and the emitted side alike, which is a property of
    the sampling point, not of this test's comparison -- the unperturbed
    initial state has no such issue for either model, so this test uses that
    instead. Measured worst-case agreement at these points, in the correct
    (unpatched) case, is machine precision (~1e-16 relative) for both models
    -- see the fix-report for the perturbed-`diagonal_jacobian`
    failing-then-restored-passing demonstration.
    """
    ode = loaded[name]
    mod = _generated_module(ode, name)
    jac = diagonal_jacobian(ode)

    np.seterr(all="ignore")
    parameters = mod["init_parameter_values"]()
    base = mod["init_state_values"]()

    dt = 1e-3
    symbolic_values = []
    emitted_values = []
    for t in (0.0, 5.0, 10.0):
        values = base.copy()
        _, captured = _call_capturing_locals(
            mod["generalized_rush_larsen"], values.copy(), t, dt, parameters
        )
        monitor = mod["monitor_values"](t, values, parameters)
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

        for x in ode.state_derivatives:
            expr = jac[x.state.name]
            if any(s not in env for s in expr.free_symbols):
                continue
            symbolic_values.append(float(sympy.lambdify(list(env), expr, "numpy")(*env.values())))
            emitted_values.append(float(captured[f"{x.name}_linearized"]))

    assert len(symbolic_values) == 3 * len(ode.states), (
        f"expected {3 * len(ode.states)} comparisons with zero skips, got {len(symbolic_values)}"
    )
    np.testing.assert_allclose(
        emitted_values,
        symbolic_values,
        atol=1e-9,
        rtol=1e-9,
        err_msg=(f"emitted d<state>_dt_linearized disagrees with diagonal_jacobian for {name}"),
    )


@pytest.mark.parametrize("name", ["tentusscher_panfilov_2006_M_cell", "ToRORd_dyn_chloride"])
def test_cse_strategies_agree_on_emitted_linearized_values(name, loaded):
    """CSE is a factoring of the same expression, so it must not change the
    result: `cse=True` and `cse=False` must emit numerically identical
    `d<state>_dt_linearized` values, at the same state/parameter vector.

    Reuses `_call_capturing_locals` (see
    `test_emitted_linearized_expression_matches_diagonal_jacobian` above) to
    read the post-CSE locals straight out of each generated module, rather
    than re-deriving expected values from `diagonal_jacobian` a second time --
    that comparison against the untouched symbolic diagonal is already made
    once, for the default, by that other test.
    """
    ode = loaded[name]
    jac = diagonal_jacobian(ode)
    linearized_names = [
        f"{x.name}_linearized" for x in ode.state_derivatives if not jac[x.state.name].is_zero
    ]
    assert linearized_names, f"expected at least one linearized state for {name}"

    np.seterr(all="ignore")
    values_by_strategy = {}
    for cse in (True, False):
        mod = _generated_module_with_cse(ode, name, cse)
        parameters = mod["init_parameter_values"]()
        base = mod["init_state_values"]()
        _, captured = _call_capturing_locals(
            mod["generalized_rush_larsen"], base.copy(), 0.0, 1e-3, parameters
        )
        values_by_strategy[cse] = np.array([captured[n] for n in linearized_names], dtype=float)

    np.testing.assert_allclose(
        values_by_strategy[False],
        values_by_strategy[True],
        atol=1e-9,
        rtol=1e-9,
        err_msg=f"cse=False disagrees with cse=True for {name}",
    )


def _linearization_block_ops(ode, cse):
    """Total `sympy.count_ops` of everything the linearization block emits at
    the given `cse` setting: every temporary's rhs, plus every state's reduced
    `d<state>_dt_linearized` expression. Mirrors what
    `test_linearization_block_stays_within_twice_the_rhs` measures for the
    default, generalized to both settings."""
    from gotranx.schemes import _linearization_plan, _taken_names

    jac = diagonal_jacobian(ode)
    to_linearize = [x for x in ode.state_derivatives if not jac[x.state.name].is_zero]
    taken = _taken_names(ode)
    plan = _linearization_plan([(x.name, jac[x.state.name]) for x in to_linearize], taken, cse)
    ops = 0
    for replacements, reduced in plan.values():
        for _, sub_expr in replacements:
            ops += sympy.count_ops(sub_expr)
        ops += sympy.count_ops(reduced)
    return ops


def test_cse_saves_operations_on_ToRORd(loaded):
    """Measured background for the default: CSE costs far fewer operations
    than inlining -- 1.08x the rhs against 9.43x, on ToRORd_dyn_chloride. That
    gap is why `cse=True` is the default and `cse=False` is reserved for the
    one thing it buys, namely no named temporaries at all. This test pins the
    direction, not the exact multiples, so it stays robust to incidental sympy
    version changes in how aggressively `cse`/`count_ops` simplify."""
    ode = loaded["ToRORd_dyn_chloride"]
    cse_ops = _linearization_block_ops(ode, True)
    inlined_ops = _linearization_block_ops(ode, False)

    assert cse_ops < inlined_ops, (cse_ops, inlined_ops)
