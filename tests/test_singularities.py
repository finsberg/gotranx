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

    Solved on the float expression as written, the root comes back 3.6e-15
    away from -23: sympy expands `-0.04*(V + 23)` to `-0.04*V - 0.92` and the
    root is `-(-0.92)/(-0.04)`. That offset is *not* harmless -- expanding a
    series about it makes `sympy.series` return 0 -- which is why
    `removable_poles` locates roots on a rationalized copy instead. This test
    only pins the helper's behavior on the raw float form.
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


def _worst_guarded_derivative_error(expr, var, center, delta):
    """Worst relative error of the guarded derivative -- series inside the
    window, direct float64 formula outside -- against a 50-digit reference,
    over a geometric sweep from 1e-9 to 0.2 on both sides of the pole."""
    import mpmath
    import numpy

    mpmath.mp.dps = 50
    replacement = singularities.taylor(expr, var, sympy.Integer(int(center)), 3)
    truth = sympy.lambdify(var, sympy.nsimplify(expr, rational=True), "mpmath")
    series = sympy.lambdify(var, sympy.diff(replacement, var), "numpy")
    direct = sympy.lambdify(var, sympy.diff(expr, var), "numpy")
    worst = 0.0
    with numpy.errstate(all="ignore"):
        for offset in (float(o) for o in numpy.geomspace(1e-9, 0.2, 40)):
            for point in (center - offset, center + offset):
                branch = series if offset < delta else direct
                got = mpmath.mpf(float(branch(numpy.float64(point))))
                expected = mpmath.diff(truth, mpmath.mpf(repr(point)))
                worst = max(worst, float(abs(got - expected) / abs(expected)))
    return worst


def test_half_width_minimizes_the_worst_case_derivative_error():
    """The linearized block uses the *derivative*, so the window must be
    placed where the guarded derivative is most accurate overall.

    Measured on ToRORd's INab kernel, worst relative error over the sweep:
    empirical crossover (delta = 0.0518) 4.8e-11; the analytic
    truncation-vs-round-off rule (delta = 0.0135) 4.2e-10; the
    value-calibrated rule (delta = 0.1) 2.1e-10. The margins below are 2x,
    not tighter: an earlier version asserted 5x against a window that sat on
    a lucky round-off cancellation (1.9e-11), which did not survive the move
    to x86-64 CI.
    """
    F_, R_, T_, P_ = 96485.0, 8314.0, 310.0, 3.75e-10
    ghk = (
        P_
        * (v * F_ * F_ / (R_ * T_))
        * (12.0 * sympy.exp(v * F_ / (R_ * T_)) - 140.0)
        / (sympy.exp(v * F_ / (R_ * T_)) - 1)
    )
    delta = singularities.half_width(ghk, v, sympy.Integer(0), 3, {})
    chosen = _worst_guarded_derivative_error(ghk, v, 0.0, delta)
    analytic = _worst_guarded_derivative_error(
        ghk, v, 0.0, singularities._analytic_half_width(ghk, v, sympy.Integer(0), 3, {})
    )
    value_calibrated = _worst_guarded_derivative_error(ghk, v, 0.0, 0.1)
    assert chosen < 1e-10, chosen
    assert chosen < analytic / 2, (chosen, analytic)
    assert chosen < value_calibrated / 2, (chosen, value_calibrated)


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


GHK_DEFAULTS = {
    F: 96485.0,
    R: 8314.0,
    T: 310.0,
    PNab: 3.75e-10,
    nai: 12.0,
    nao: 140.0,
    v: -80.0,
}


def test_removable_poles_finds_the_ghk_pole_and_nothing_else():
    poles = singularities.removable_poles(GHK, GHK_DEFINITIONS, frozenset({v}), GHK_DEFAULTS)
    assert len(poles) == 1
    assert poles[0].var == v
    # `value` is emitted as a float so the generated condition reads
    # naturally, so compare as one: sympy's Float(0) is not Integer(0).
    assert float(poles[0].value) == 0.0
    assert poles[0].half_width > 0


def test_removable_poles_skips_a_genuine_pole():
    IpCa = GpCa * cai / (KmCap + cai)
    poles = singularities.removable_poles(
        IpCa, {}, frozenset({cai}), {GpCa: 0.0005, KmCap: 0.0005, cai: 1e-4}
    )
    assert poles == ()


def test_removable_poles_skips_a_location_that_is_not_a_real_number():
    """tentusscher_panfilov's Ca_i buffering root is
    -K_buf_c +- sqrt(-Buf_c*K_buf_c), imaginary for positive parameters. A
    guard window needs a real center."""
    Buf, K = sympy.symbols("Buf K", real=True)
    expr = cai / (cai**2 + 2 * K * cai + K**2 + Buf * K)
    poles = singularities.removable_poles(
        expr, {}, frozenset({cai}), {Buf: 0.2, K: 0.001, cai: 1e-4}
    )
    assert poles == ()


def test_removable_poles_skips_an_expression_with_a_constant_denominator():
    """The pre-filter: a denominator that is a parameter or a literal cannot
    vanish for a state-dependent reason, so the cone is never inlined."""
    C = sympy.Symbol("C", real=True)
    assert singularities.removable_poles((v + 3) / C, {}, frozenset({v}), {C: 1.0, v: -80.0}) == ()


def test_guard_emits_a_piecewise_on_the_window():
    pole = singularities.RemovablePole(
        var=V,
        value=sympy.Integer(-10),
        replacement=sympy.Integer(10),
        half_width=1e-2,
        rewritten=V,
    )
    guarded = singularities.guard(V, (pole,))
    assert isinstance(guarded, sympy.Piecewise)
    assert float(guarded.subs(V, -10)) == 10.0
    assert float(guarded.subs(V, 5)) == 5.0


def test_rewrite_is_the_identity_when_there_is_no_pole():
    expr = sympy.exp(V) + 3
    assert singularities.rewrite(expr, {}, frozenset({V}), {V: 0.0}) is expr


def test_rewrite_guards_the_ghk_expression_in_the_state_variable():
    """The guard must be written in v, not vfrt. The forward-mode sweep in
    linearization.py is seeded at v, so a branch written in vfrt would
    differentiate to zero there -- the same failure the order-0 replacement
    had."""
    import numpy

    guarded = singularities.rewrite(GHK, GHK_DEFINITIONS, frozenset({v}), GHK_DEFAULTS)
    assert v in guarded.free_symbols
    assert VFRT not in guarded.free_symbols
    assert VFFRT not in guarded.free_symbols

    numeric = sympy.lambdify(
        v,
        guarded.xreplace(
            {
                symbol: sympy.Float(value)
                for symbol, value in GHK_DEFAULTS.items()
                if symbol is not v
            }
        ),
        "numpy",
    )
    numpy.seterr(all="ignore")
    assert numpy.isfinite(float(numeric(numpy.float64(0.0))))


def test_default_values_resolves_intermediates_in_order():
    """Intermediates that do not depend on the pole variable stay opaque
    symbols in the guarded expression (see `inline`), so the window
    half-width and the numeric check need a value for them as well."""
    y, z = sympy.symbols("y z", real=True)
    values = singularities.default_values({y: 2 * a, z: y + 1}, {a: 3.0})
    assert values[y] == 6.0
    assert values[z] == 7.0
    assert values[a] == 3.0


def test_default_values_skips_what_cannot_be_evaluated():
    """ToRORd's bundled state has CaTrpn = 0 and an intermediate comparing
    CaTrpn**(-ntm/2) against 100, which raises under exact arithmetic. That
    must leave the intermediate without a default, not abort the model."""
    y = sympy.Symbol("y", real=True)
    expr = sympy.Piecewise((a**-2, a**-2 < 100), (100, True))
    values = singularities.default_values({y: expr}, {a: 0.0})
    assert y not in values


def test_removable_poles_guards_through_an_opaque_intermediate():
    """ToRORd's ICab keeps `gamma_cai` as a symbol after inlining in v; the
    guard must still be emitted when its default is known."""
    definitions = dict(GHK_DEFINITIONS)
    definitions[gamma] = sympy.exp(sympy.sqrt(cai))
    base = dict(GHK_DEFAULTS)
    base[cai] = 1e-4
    defaults = singularities.default_values(definitions, base)
    poles = singularities.removable_poles(gamma * GHK, definitions, frozenset({v, cai}), defaults)
    assert [str(p.var) for p in poles] == ["v"]


def test_rewrite_guards_a_gate_rate_written_through_an_intermediate():
    """`xreplace` rebuilds with evaluate=True, which folds
    `exp(-0.04*(V + 23))` into `0.398519041084514*exp(-0.04*V)`. Once folded
    the pole is no longer exactly at -23 in any arithmetic, so inlining must
    not fold. The replacement's value and slope at the pole are checked
    against the analytic limits 0.2/0.04 = 5 and 0.2/2 = 0.1.
    """
    shifted = sympy.Symbol("shifted", real=True)
    with sympy.evaluate(False):
        definitions = {shifted: V + 23}
        rate = 0.2 * shifted / (1 - sympy.exp(-0.04 * shifted))
    guarded = singularities.rewrite(rate, definitions, frozenset({V}), {V: -80.0})
    assert guarded.has(sympy.Piecewise), "the pole was not guarded"
    replacement = guarded.args[0][0]
    assert abs(float(replacement.subs(V, -23)) - 5.0) < 1e-10
    assert abs(float(sympy.diff(replacement, V).subs(V, -23)) - 0.1) < 1e-9


def test_behavior_carried_over_from_the_previous_mechanism():
    """The cases `tests/test_atoms.py::test_singularities` pinned for the
    mechanism this module replaces, restated against the new one."""
    b = sympy.Symbol("b", real=True)
    states = frozenset({x})
    defaults = {a: 1.0, b: 2.0, x: 1.0}

    def poles(expr):
        return singularities.removable_poles(expr, {}, states, defaults)

    (z,) = poles(x / (sympy.exp(x) - 1.0))
    assert float(z.value) == 0.0
    assert abs(float(z.replacement.subs(x, 0)) - 1.0) < 1e-12

    (z1,) = poles(x / (sympy.exp(x) - 1.0) + a)
    assert abs(float(z1.replacement.subs({x: 0, a: 1.0})) - 2.0) < 1e-12

    z2 = poles(x / (sympy.exp(x) - 1.0) + a + (x - 2) / (sympy.exp(x) - sympy.exp(2)))
    assert sorted(float(p.value) for p in z2) == [0.0, 2.0]

    assert poles(a / b) == ()  # denominator is not state-dependent
    assert poles(x / b) == ()  # denominator is not state-dependent
    assert poles(b / x) == ()  # a genuine pole, not a removable one


def test_factor_roots_treats_a_constant_exponential_as_a_constant():
    """`exp(2)` contains an `exp` but not the variable; it is the B in
    A*exp(u) + B."""
    assert singularities.factor_roots(sympy.exp(x) - sympy.exp(2), x) == [2]


def test_denominator_factors_does_not_look_inside_a_denominator():
    """Only outermost denominators are scanned. Looking inside finds the
    buffering terms' removable singularities at negative concentrations,
    triples ToRORd's guarded set, and does not terminate on ORdmm_Land."""
    K, c = sympy.symbols("K c", real=True)
    inner = K + x
    factors = singularities.denominator_factors(1 / (1 + c / inner**2))
    assert factors == (1 + c / inner**2,)
    assert inner not in factors


def test_removable_poles_skips_an_expression_too_large_to_expand():
    """sympy.series has no bound of its own; on ToRORd's 267-operation E1_i it
    did not finish in 100 s."""
    big = sum(sympy.sin(k * x) ** k for k in range(1, 60))
    expr = big * x / (sympy.exp(x) - 1)
    assert sympy.count_ops(expr) > singularities.MAX_SERIES_OPS
    assert singularities.removable_poles(expr, {}, frozenset({x}), {x: 1.0}) == ()


def test_pole_screen_rejects_a_genuine_pole_but_never_a_removable_one():
    """The numeric screen only ever rejects, so the failure that matters is a
    false rejection -- which would silently drop a guard."""
    IpCa = GpCa * cai / (KmCap + cai)
    assert singularities._looks_like_a_genuine_pole(
        IpCa, cai, -KmCap, {GpCa: 0.0005, KmCap: 0.0005}
    )
    ghk = singularities._exact(singularities.inline(GHK, GHK_DEFINITIONS, v))
    assert not singularities._looks_like_a_genuine_pole(ghk, v, sympy.Integer(0), GHK_DEFAULTS)
    gate = singularities._exact((V + 10) / (sympy.exp((V + 10) / 10) - 1))
    assert not singularities._looks_like_a_genuine_pole(gate, V, sympy.Integer(-10), {})


def _exp_with_ulp_errors(seed):
    """An `exp` that is off by -1, 0 or +1 ulp, deterministically per input.

    Models what differs between CI hosts: numpy's own SIMD `exp` on x86-64 is
    not correctly rounded, while glibc's (which numpy uses on aarch64) is.
    Works for both float64 and 53-bit mpmath arguments.
    """
    import zlib

    import mpmath
    import numpy

    def exp(x):
        key = zlib.crc32(repr(x).encode() + bytes([seed])) % 3
        if isinstance(x, mpmath.mpf):
            y = mpmath.exp(x)
            return y if key == 0 else y * (1 + (1 if key == 1 else -1) * mpmath.mpf(2) ** -52)
        y = numpy.exp(x)
        return y if key == 0 else numpy.nextafter(y, numpy.inf if key == 1 else -numpy.inf)

    return exp


def test_half_width_does_not_depend_on_one_ulp_differences_in_exp(monkeypatch):
    """The window half-width is emitted into generated code, so it must not
    depend on the host's floating-point `exp`. It did: CI on x86-64 chose
    0.0518 for this kernel where aarch64 chose 0.0373, because the search
    picked a single grid point where float64 round-off happened to cancel,
    and a one-ulp difference in `exp` moves that cancellation."""
    F_, R_, T_, P_ = 96485.0, 8314.0, 310.0, 3.75e-10
    ghk = (
        P_
        * (v * F_ * F_ / (R_ * T_))
        * (12.0 * sympy.exp(v * F_ / (R_ * T_)) - 140.0)
        / (sympy.exp(v * F_ / (R_ * T_)) - 1)
    )
    reference = singularities.half_width(ghk, v, sympy.Integer(0), 3, {})

    original = sympy.lambdify
    for seed in range(6):

        def noisy_lambdify(args, expr, modules=None, seed=seed, **kwargs):
            return original(args, expr, [{"exp": _exp_with_ulp_errors(seed)}, modules], **kwargs)

        monkeypatch.setattr(sympy, "lambdify", noisy_lambdify)
        assert singularities.half_width(ghk, v, sympy.Integer(0), 3, {}) == reference, seed
        monkeypatch.setattr(sympy, "lambdify", original)
