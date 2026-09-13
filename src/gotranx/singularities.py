r"""Removable singularities in model expressions.

A gate rate like :math:`x/(e^x - 1)` has a removable pole at :math:`x = 0`.
Evaluated in float64 it loses all precision as :math:`x \to 0` and raises at
exactly 0; *differentiated* -- which is what the Rush-Larsen linearized block
does -- it has a double pole, and the wrong value it produces is O(1) and can
have the wrong sign.

This module finds such poles and replaces a neighbourhood of each with a
truncated Taylor series. It is sympy-only: it knows nothing about
``Assignment``, lark, or ``ODE``, so both the model layer and the schemes can
use it.

Two sympy entry points are deliberately *not* used here:

``sympy.limit``
    Silently returns 0 for float-coefficient gate rates, e.g.
    ``limit(0.2*(V + 23)/(1 - exp(-0.04*(V + 23))), V, -23)`` is 0 where the
    true limit is 5. ``.ode`` files are written with float coefficients, so
    this is not a corner case.

``sympy.singularities`` / ``sympy.solve``
    Neither terminates on inlined model expressions. Measured on
    ``ToRORd_dyn_chloride``: ``singularities`` fails to finish within 10 s for
    37 of 168 (assignment, state) pairs, and ``solve`` fails to finish within
    5 s on a 4-operation denominator such as ``exp(-(v + a)/b) + 1``, which has
    no real root at all. Pole *locations* are therefore found structurally, by
    matching the handful of denominator shapes that admit a closed-form root.
"""

from __future__ import annotations

from typing import Mapping

import sympy

__all__ = [
    "EPSILON",
    "MAX_HALF_WIDTH",
    "MAX_INLINE_OPS",
    "MIN_HALF_WIDTH",
    "agrees_numerically",
    "denominator_factors",
    "factor_roots",
    "half_width",
    "inline",
    "is_removable",
    "taylor",
]

#: Exceptions the sympy series machinery raises on an expression it cannot
#: expand. Caught rather than propagated: a pole we cannot classify is simply
#: left unguarded, which is what gotranx did for every pole before this module
#: existed.
_SERIES_ERRORS = (
    ValueError,
    NotImplementedError,
    TypeError,
    AttributeError,
    sympy.PoleError,
    RecursionError,
)

MIN_HALF_WIDTH = 1e-8
"""Narrowest guard window allowed, in the guarded variable's own units."""

MAX_HALF_WIDTH = 1e-1
"""Widest guard window allowed, in the guarded variable's own units."""

EPSILON = 2.220446049250313e-16
"""float64 machine epsilon, the round-off scale the window is balanced against."""

MAX_INLINE_OPS = 5000
"""Give up inlining past this many operations.

ToRORd_dyn_chloride's largest candidate reaches 939 operations, so this is
loose. It exists to bound the worst case on a model nobody has tried yet, not
to exclude anything in the bundled set.
"""


def denominator_factors(expr: sympy.Expr) -> tuple[sympy.Expr, ...]:
    """The factors of ``expr``'s denominator.

    Parameters
    ----------
    expr : sympy.Expr
        Any expression.

    Returns
    -------
    tuple[sympy.Expr, ...]
        The factors of the denominator of ``sympy.together(expr)``. An
        expression with no denominator yields ``(1,)``.
    """
    _, den = sympy.fraction(sympy.together(expr))
    return tuple(sympy.Mul.make_args(sympy.factor_terms(den)))


def _linear_root(expr: sympy.Expr, var: sympy.Symbol) -> sympy.Expr | None:
    """Root of ``expr == 0`` when ``expr`` is linear in ``var``, else None."""
    try:
        poly = sympy.Poly(expr, var)
    except (sympy.PolynomialError, sympy.GeneratorsNeeded):
        return None
    if poly.degree() != 1:
        return None
    slope, intercept = poly.all_coeffs()
    if slope == 0:
        return None
    return sympy.simplify(-intercept / slope)


def factor_roots(
    factor: sympy.Expr,
    var: sympy.Symbol,
    max_degree: int = 2,
) -> list[sympy.Expr]:
    r"""Real roots of a denominator ``factor`` in ``var``, found structurally.

    Only shapes with a closed-form root are matched, because the general
    solvers do not terminate on inlined model expressions (see the module
    docstring):

    * ``A*exp(u) + B`` with ``A`` and ``B`` free of ``var`` and ``u`` linear in
      ``var``. Real only when ``-B/A > 0``, giving ``u = log(-B/A)``. This is
      every GHK denominator (``exp(vfrt) - 1``) and every gate-rate
      denominator (``1 - exp(-0.04*(V + 23))``).
    * a polynomial in ``var`` of degree at most ``max_degree``.

    Anything else returns an empty list: the pole, if there is one, is left
    unguarded.

    Parameters
    ----------
    factor : sympy.Expr
        One factor of a denominator.
    var : sympy.Symbol
        The variable to solve in.
    max_degree : int, optional
        Highest polynomial degree to solve, by default 2.

    Returns
    -------
    list[sympy.Expr]
        Candidate root locations, possibly in terms of parameters. Roots sympy
        can prove non-real are dropped; roots it cannot decide are kept here
        and filtered numerically later.
    """
    if var not in factor.free_symbols:
        return []

    if not factor.has(sympy.exp):
        try:
            poly = sympy.Poly(factor, var)
        except (sympy.PolynomialError, sympy.GeneratorsNeeded):
            return []
        if poly.degree() > max_degree:
            return []
        return [root for root in sympy.roots(poly) if root.is_real is not False]

    # A*exp(u) + B
    if not isinstance(factor, sympy.Add) or len(factor.args) != 2:
        return []
    exponentials = [term for term in factor.args if term.has(sympy.exp)]
    constants = [term for term in factor.args if not term.has(sympy.exp)]
    if len(exponentials) != 1 or len(constants) != 1:
        return []

    intercept = constants[0]
    coefficient: sympy.Expr = sympy.S.One
    exponent: sympy.Expr | None = None
    for term in sympy.Mul.make_args(exponentials[0]):
        if isinstance(term, sympy.exp):
            if exponent is not None:
                return []  # a product of exponentials, not this shape
            exponent = term.args[0]
        else:
            coefficient = coefficient * term
    if exponent is None:
        return []
    if var in coefficient.free_symbols or var in intercept.free_symbols:
        return []

    ratio = sympy.simplify(-intercept / coefficient)
    if ratio.is_positive is not True:
        return []
    root = _linear_root(exponent - sympy.log(ratio), var)
    return [] if root is None else [root]


def inline(
    expr: sympy.Expr,
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    var: sympy.Symbol,
) -> sympy.Expr | None:
    """Rewrite ``expr`` in terms of ``var``, expanding only what depends on it.

    A pole's removability is only visible after inlining: ToRORd's ``INab`` is
    ``PNab*vffrt*(...)/(exp(vfrt) - 1)``, and it is ``vffrt = v*F*F/(R*T)``
    that makes the numerator vanish as ``v -> 0``. Held as a separate named
    symbol, that vanishing factor is invisible and the removable pole looks
    essential -- which is why gotranx discarded every GHK pole as infinite
    before this module existed.

    Only intermediates that *transitively depend on* ``var`` are expanded.
    Inlining the rest as well costs a great deal and buys nothing: on ToRORd,
    expanding ``PhiCaL_ss``'s whole cone pulls in the ionic-strength terms
    ``gamma_cass`` and ``gamma_cao``, which do not mention ``v``, and grows the
    generated rhs from 32,499 to 74,662 characters and scheme generation from
    2.75 s to 130.68 s. Restricted to ``v``-dependent intermediates the same
    guard costs 35,766 characters and 4.10 s.

    Leaving a ``var``-independent intermediate in place is also correct for the
    forward-mode sweep in :mod:`gotranx.linearization`: it carries its own
    tangent, exactly as it does in an unguarded assignment.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to rewrite.
    definitions : Mapping[sympy.Symbol, sympy.Expr]
        Every intermediate symbol in the model and its defining expression.
    var : sympy.Symbol
        The state variable the pole lives in.

    Returns
    -------
    sympy.Expr | None
        The rewritten expression, or None if it grew past
        :data:`MAX_INLINE_OPS`.
    """
    dependent = _depending_on(definitions, {var})
    while True:
        substitutions = {
            symbol: definitions[symbol] for symbol in expr.free_symbols if symbol in dependent
        }
        if not substitutions:
            return expr
        expr = expr.xreplace(substitutions)
        if sympy.count_ops(expr) > MAX_INLINE_OPS:
            return None


def _depending_on(
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    seeds: set[sympy.Symbol],
) -> set[sympy.Symbol]:
    """Symbols in ``definitions`` that transitively depend on any of ``seeds``.

    ``definitions`` is not guaranteed to be topologically sorted, so this
    iterates to a fixed point rather than assuming one pass suffices.
    """
    dependent: set[sympy.Symbol] = set()
    changed = True
    while changed:
        changed = False
        for symbol, definition in definitions.items():
            if symbol in dependent:
                continue
            if (definition.free_symbols & seeds) or (definition.free_symbols & dependent):
                dependent.add(symbol)
                changed = True
    return dependent


def is_removable(expr: sympy.Expr, var: sympy.Symbol, value: sympy.Expr) -> bool:
    """Whether ``expr``'s singularity at ``var = value`` is removable.

    Decided by the leading exponent of the Laurent expansion: removable means
    no negative powers of ``var - value``.

    Testing instead whether the truncated series contains ``oo``, ``zoo`` or
    ``nan`` does not work, because the Laurent expansion around a *genuine*
    simple pole contains none of them -- it contains a ``1/(var - value)``
    term. Measured on ToRORd, that test classifies all 40 pole candidates as
    removable, including ``IpCa`` at ``cai = -KmCap``, whose "replacement"
    would be ``-GpCa*KmCap/(KmCap + cai) + GpCa``.

    Parameters
    ----------
    expr : sympy.Expr
        The expression, already inlined in ``var``.
    var : sympy.Symbol
        The variable the pole lives in.
    value : sympy.Expr
        Where the pole is.

    Returns
    -------
    bool
        True if the singularity is removable. False when sympy cannot decide,
        which leaves the pole unguarded rather than guarded wrongly.
    """
    offset = sympy.Dummy("offset", positive=True)
    try:
        _, exponent = expr.subs(var, value + offset).leadterm(offset)
    except _SERIES_ERRORS:
        return False
    return bool(exponent >= 0)


def taylor(
    expr: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    order: int,
) -> sympy.Expr | None:
    """Truncated Taylor series of ``expr`` about ``var = value``.

    ``order`` must be at least 1. A constant replacement -- the limit, which is
    what gotranx emitted before this module existed -- differentiates to zero,
    so an assignment guarded that way contributes nothing to the Jacobian
    diagonal and the linearization is silently wrong. With order >= 1 the
    guard differentiates correctly and
    :func:`gotranx.linearization.diagonal_jacobian` needs no singularity
    awareness of its own.

    Parameters
    ----------
    expr : sympy.Expr
        The expression, already inlined in ``var``.
    var : sympy.Symbol
        The variable to expand in.
    value : sympy.Expr
        The point to expand about.
    order : int
        Highest power of ``var - value`` to keep. Order 3 costs the same as
        order 2 for the canonical gate-rate kernel, whose third Bernoulli
        number is zero.

    Returns
    -------
    sympy.Expr | None
        The expanded polynomial, or None if sympy could not produce a finite
        series.

    Raises
    ------
    ValueError
        If ``order`` is less than 1.
    """
    if order < 1:
        raise ValueError(f"order must be at least 1, got {order}")
    try:
        series = sympy.series(expr, var, value, order + 1).removeO()
    except _SERIES_ERRORS:
        return None
    if series.has(sympy.oo, -sympy.oo, sympy.zoo, sympy.nan):
        return None
    return sympy.expand(series)


def _numeric(expr: sympy.Expr, defaults: Mapping[sympy.Symbol, float]) -> float | None:
    """``expr`` as a real, finite float at default values, or None.

    None means "not usable as a guard location or as a check point": complex,
    infinite, NaN, or still carrying a free symbol ``defaults`` does not cover.
    """
    substituted = expr.xreplace({symbol: sympy.Float(x) for symbol, x in defaults.items()})
    try:
        value = complex(substituted.evalf())
    except (TypeError, ValueError, AttributeError):
        return None
    if value != value or abs(value.real) == float("inf") or abs(value.imag) == float("inf"):
        return None
    if abs(value.imag) > 1e-12 * max(1.0, abs(value.real)):
        return None
    return value.real


def _series_coefficients(
    expr: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    upto: int,
    defaults: Mapping[sympy.Symbol, float],
) -> list[float] | None:
    """Numeric Taylor coefficients a_0 .. a_upto of ``expr`` about ``value``."""
    offset = sympy.Symbol("_offset")
    try:
        series = sympy.series(expr.subs(var, value + offset), offset, 0, upto + 1).removeO()
        poly = sympy.Poly(sympy.expand(series), offset)
    except _SERIES_ERRORS + (sympy.PolynomialError, sympy.GeneratorsNeeded):
        return None
    coefficients = []
    for k in range(upto + 1):
        coefficient = _numeric(poly.coeff_monomial(offset**k), defaults)
        if coefficient is None:
            return None
        coefficients.append(coefficient)
    return coefficients


def half_width(
    expr: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    order: int,
    defaults: Mapping[sympy.Symbol, float],
) -> float | None:
    r"""Half-width of the window in which the Taylor branch is used.

    Placed where the series' truncation error crosses the direct formula's
    float64 round-off, measured on the *derivative* -- the quantity the
    Rush-Larsen linearized block needs. With coefficients :math:`a_k` and
    :math:`k` the first nonzero index above ``order``, truncation of the
    derivative goes like :math:`k |a_k| \delta^{k-1} / |a_1|` while round-off
    of the direct derivative grows like :math:`\epsilon/\delta^2`, so the two
    cross at

    .. math::
        \delta^{k+1} = \frac{\epsilon |a_1|}{k |a_k|}

    clamped to ``[MIN_HALF_WIDTH, MAX_HALF_WIDTH]``. The clamp matters at both
    ends: the lower bound keeps the window wider than the region where the
    direct formula is already catastrophically wrong (222% relative error at
    1e-8 for the canonical kernel), and the upper bound keeps the series out of
    the region where the direct formula is the more accurate of the two.

    Calibrating on the *value* instead -- bounding
    :math:`|a_k \delta^k| \le \mathrm{tol}|a_0|` -- is one order too generous,
    because differentiating a truncated series loses an order. On ToRORd's
    ``INab`` that rule returns 0.1, where the series derivative is 4.9e-10 off
    a 50-digit reference and the direct float64 derivative is 3.0e-13 off.

    The rule is scale-free: it adapts to a gate rate written ``-0.1*(V + 47)``
    as readily as to one written ``(V + 10)/10``, where a fixed window in the
    guarded variable's own units would not.

    Parameters
    ----------
    expr : sympy.Expr
        The expression, already inlined in ``var``.
    var : sympy.Symbol
        The variable the pole lives in.
    value : sympy.Expr
        Where the pole is.
    order : int
        The order of the Taylor replacement.
    defaults : Mapping[sympy.Symbol, float]
        Default values for parameters and other states, used to make the
        coefficients numeric. A window half-width has to be a number.

    Returns
    -------
    float | None
        The half-width, or None if the coefficients could not be evaluated.
    """
    coefficients = _series_coefficients(expr, var, value, order + 8, defaults)
    if coefficients is None:
        return None

    tail = [(k, c) for k, c in enumerate(coefficients) if k > order and c != 0.0]
    if not tail:
        # No truncation error within reach: as far as we can see the
        # replacement is exact, so use the widest window allowed.
        return MAX_HALF_WIDTH
    k, a_k = tail[0]

    # a_1 is the scale the derivative's *relative* error is measured against.
    # If the derivative vanishes at the pole, fall back to the value's scale.
    reference = coefficients[1] if coefficients[1] != 0.0 else coefficients[0]
    if reference == 0.0:
        return None
    delta = (EPSILON * abs(reference) / (k * abs(a_k))) ** (1.0 / (k + 1))
    return min(max(delta, MIN_HALF_WIDTH), MAX_HALF_WIDTH)


def agrees_numerically(
    expr: sympy.Expr,
    replacement: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    delta: float,
    defaults: Mapping[sympy.Symbol, float],
    tolerance: float = 1e-6,
) -> bool:
    """Spot-check a replacement against the expression it replaces.

    Symbolic tools have silently produced wrong answers twice in this design's
    investigation -- ``sympy.limit`` returning 0 on a float-coefficient gate
    rate, and a Laurent branch passing an ``oo``/``zoo``/``nan`` scan -- so
    every replacement is checked numerically before it is emitted. A guard
    that fails is dropped, not emitted.

    The check is made at ``value +- delta``, the window *edge*, not at its
    centre. ``delta`` is placed where the two branches cross over, so both are
    accurate there; closer in, the direct formula is the inaccurate one and a
    disagreement would say nothing about the replacement.

    Parameters
    ----------
    expr : sympy.Expr
        The original expression.
    replacement : sympy.Expr
        The proposed Taylor replacement.
    var : sympy.Symbol
        The variable the pole lives in.
    value : sympy.Expr
        Where the pole is.
    delta : float
        The window half-width.
    defaults : Mapping[sympy.Symbol, float]
        Default values for parameters and other states.
    tolerance : float, optional
        Largest acceptable relative disagreement, by default 1e-6.

    Returns
    -------
    bool
        True if the replacement may be emitted.
    """
    centre = _numeric(value, defaults)
    if centre is None:
        return False
    for point in (centre - delta, centre + delta):
        substitution = {var: sympy.Float(point)}
        original = _numeric(expr.subs(substitution), defaults)
        proposed = _numeric(replacement.subs(substitution), defaults)
        if original is None or proposed is None:
            return False
        scale = max(abs(original), abs(proposed), 1e-300)
        if abs(original - proposed) / scale > tolerance:
            return False
    return True
