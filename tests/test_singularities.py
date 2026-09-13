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
