"""Unit tests for `gotranx.singularities`."""

from __future__ import annotations

import sympy

from gotranx import singularities

V = sympy.Symbol("V", real=True)
x = sympy.Symbol("x", real=True)
a = sympy.Symbol("a", real=True)


def test_denominator_factors_splits_a_product():
    expr = (V + 1) / ((sympy.exp(V) - 1) * (V - 3))
    factors = singularities.denominator_factors(expr)
    assert sympy.exp(V) - 1 in factors
    assert V - 3 in factors


def test_denominator_factors_is_empty_for_a_polynomial():
    assert singularities.denominator_factors(V**2 + 3 * V) == (sympy.S.One,)


def test_factor_roots_finds_the_ghk_root():
    """`exp(u) - 1` vanishes where u = 0; u = V*F/(R*T) is linear in V."""
    F, R, T = sympy.symbols("F R T", real=True)
    assert singularities.factor_roots(sympy.exp(V * F / (R * T)) - 1, V) == [0]


def test_factor_roots_finds_a_gate_rate_root_with_float_coefficients():
    """The shape .ode files actually use -- see the design document's F5.

    The root comes back as -23 to within float round-off rather than exactly:
    sympy expands `-0.04*(V + 23)` to `-0.04*V - 0.92` and the root is
    `-(-0.92)/(-0.04)`, which is 3.6e-15 away from -23 in float64. That is
    immaterial next to a guard window of ~1e-2, so this asserts closeness
    rather than pretending the arithmetic is exact.
    """
    roots = singularities.factor_roots(1 - sympy.exp(-0.04 * (V + 23)), V)
    assert len(roots) == 1
    assert abs(float(roots[0]) + 23) < 1e-12


def test_factor_roots_finds_a_scaled_gate_rate_root():
    assert singularities.factor_roots(sympy.exp((V + 10) / 10) - 1, V) == [-10]


def test_factor_roots_rejects_a_denominator_with_no_real_root():
    """`exp(u) + 1` is never zero for real u. sympy.solve spends more than 5 s
    discovering this; the structural matcher answers immediately."""
    assert singularities.factor_roots(sympy.exp(-0.1 * (V + 47)) + 1, V) == []


def test_factor_roots_finds_a_linear_polynomial_root():
    assert singularities.factor_roots(x + a, x) == [-a]


def test_factor_roots_skips_a_high_degree_polynomial():
    assert singularities.factor_roots(x**5 + a, x) == []


def test_factor_roots_skips_a_factor_free_of_the_variable():
    assert singularities.factor_roots(sympy.exp(a) - 1, V) == []


F, R, T, PNab, nai, nao = sympy.symbols("F R T PNab nai nao", real=True)
v = sympy.Symbol("v", real=True)
cai, KmCap, GpCa = sympy.symbols("cai KmCap GpCa", real=True)
gamma = sympy.Symbol("gamma", real=True)

VFRT = sympy.Symbol("vfrt", real=True)
VFFRT = sympy.Symbol("vffrt", real=True)
GHK_DEFINITIONS = {VFRT: v * F / (R * T), VFFRT: v * F * F / (R * T)}
GHK = PNab * VFFRT * (nai * sympy.exp(VFRT) - nao) / (sympy.exp(VFRT) - 1)


def test_inline_expands_only_what_depends_on_the_variable():
    """`gamma` does not depend on v, so it must stay an opaque symbol --
    inlining it is what made the generated ToRORd rhs 2.3x larger."""
    definitions = dict(GHK_DEFINITIONS)
    definitions[gamma] = sympy.exp(sympy.sqrt(cai))
    inlined = singularities.inline(gamma * GHK, definitions, v)
    assert gamma in inlined.free_symbols
    assert VFRT not in inlined.free_symbols
    assert VFFRT not in inlined.free_symbols
    assert v in inlined.free_symbols


def test_inline_gives_up_past_the_operation_budget():
    """Ten distinct 600-operation summands cannot collapse into each other, so
    inlining them really does grow the tree past the budget. (A single
    definition used twice would not: `big*big` is `big**2`, one node.)"""
    parts = {
        sympy.Symbol(f"part{i}", real=True): sum(sympy.sin(i * k * v) ** k for k in range(1, 200))
        for i in range(10)
    }
    expr = sum(parts)
    assert sympy.count_ops(expr) < singularities.MAX_INLINE_OPS
    assert singularities.inline(expr, parts, v) is None


def test_is_removable_accepts_the_ghk_pole():
    inlined = singularities.inline(GHK, GHK_DEFINITIONS, v)
    assert singularities.is_removable(inlined, v, sympy.Integer(0))


def test_is_removable_rejects_a_genuine_simple_pole():
    """ToRORd's IpCa at cai = -KmCap. Deciding removability by scanning the
    truncated series for oo/zoo/nan calls this removable and would emit
    -GpCa*KmCap/(KmCap + cai) + GpCa as the replacement -- itself infinite at
    exactly the guarded point."""
    IpCa = GpCa * cai / (KmCap + cai)
    assert not singularities.is_removable(IpCa, cai, -KmCap)


def test_is_removable_rejects_a_double_pole():
    assert not singularities.is_removable(1 / (v - 3) ** 2, v, sympy.Integer(3))


def test_taylor_of_a_float_coefficient_gate_rate_is_right():
    """Design document F5: sympy.limit returns 0 here. The series does not."""
    rate = 0.2 * (V + 23) / (1 - sympy.exp(-0.04 * (V + 23)))
    replacement = singularities.taylor(rate, V, sympy.Integer(-23), 3)
    assert abs(float(replacement.subs(V, -23)) - 5.0) < 1e-12


def test_taylor_of_the_toy_gate_rate_is_the_expected_polynomial():
    """Design document F4: order >= 1 is what makes differentiation through
    the guard correct. Order 0 (`10`) differentiates to zero, which is the
    entire bug."""
    rate = (V + 10) / (sympy.exp((V + 10) / 10) - 1)
    replacement = singularities.taylor(rate, V, sympy.Integer(-10), 3)
    assert sympy.simplify(replacement - (V**2 / 120 - V / 3 + sympy.Rational(35, 6))) == 0
    assert float(replacement.subs(V, -10)) == 10.0
    assert float(sympy.diff(replacement, V).subs(V, -10)) == -0.5


def test_half_width_lands_near_the_crossover_for_the_canonical_kernel():
    """The design document's tolerance table puts the crossover at ~1e-3 in
    the local variable. Here the local variable is (V + 10)/10, so ~1e-2 in V."""
    rate = (V + 10) / (sympy.exp((V + 10) / 10) - 1)
    delta = singularities.half_width(rate, V, sympy.Integer(-10), 3, {})
    assert 1e-3 < delta < 5e-2, delta


def test_half_width_is_clamped_above():
    """A pure polynomial has no truncation error at all; the window must not
    grow without bound."""
    delta = singularities.half_width(V**2 + 1, V, sympy.Integer(0), 3, {})
    assert delta == singularities.MAX_HALF_WIDTH


def test_half_width_beats_the_direct_formula_on_its_own_derivative():
    """The linearized block uses the *derivative*, so that is what the window
    must be calibrated on. Calibrating on the value returns 0.1 here, where
    the series derivative is 4.9e-10 off a 50-digit reference and the direct
    float64 derivative is only 3.0e-13 off -- a guard three orders of
    magnitude worse than the formula it replaces."""
    import mpmath

    mpmath.mp.dps = 50
    F_, R_, T_, P_ = 96485.0, 8314.0, 310.0, 3.75e-10
    ghk = (
        P_
        * (v * F_ * F_ / (R_ * T_))
        * (12.0 * sympy.exp(v * F_ / (R_ * T_)) - 140.0)
        / (sympy.exp(v * F_ / (R_ * T_)) - 1)
    )
    delta = singularities.half_width(ghk, v, sympy.Integer(0), 3, {})
    replacement = singularities.taylor(ghk, v, sympy.Integer(0), 3)

    exact = sympy.lambdify(v, sympy.diff(sympy.nsimplify(ghk, rational=True), v), "mpmath")
    series = sympy.lambdify(v, sympy.diff(replacement, v), "mpmath")
    worst = max(
        abs(series(mpmath.mpf(delta * f)) - exact(mpmath.mpf(delta * f)))
        / abs(exact(mpmath.mpf(delta * f)))
        for f in (1.0, 0.5, 0.1, 0.01)
    )
    assert float(worst) < 1e-11, float(worst)


def test_agrees_numerically_accepts_a_correct_replacement():
    rate = 0.2 * (V + 23) / (1 - sympy.exp(-0.04 * (V + 23)))
    replacement = singularities.taylor(rate, V, sympy.Integer(-23), 3)
    assert singularities.agrees_numerically(rate, replacement, V, sympy.Integer(-23), 1e-2, {})


def test_agrees_numerically_rejects_the_limit_value_sympy_gets_wrong():
    """Design document F5: sympy.limit returns 0 for this rate, where the
    truth is 5. The numeric spot check is the backstop that catches exactly
    this class of silent symbolic failure."""
    rate = 0.2 * (V + 23) / (1 - sympy.exp(-0.04 * (V + 23)))
    assert not singularities.agrees_numerically(
        rate, sympy.Integer(0), V, sympy.Integer(-23), 1e-2, {}
    )
