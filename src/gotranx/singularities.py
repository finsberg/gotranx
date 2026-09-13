r"""Removable singularities in model expressions.

A gate rate like :math:`x/(e^x - 1)` has a removable pole at :math:`x = 0`.
Evaluated in float64 it loses all precision as :math:`x \to 0` and raises at
exactly 0; *differentiated* -- which is what the Rush-Larsen linearized block
does -- it has a double pole, and the wrong value it produces is O(1) and can
have the wrong sign.

This module finds such poles and replaces a neighborhood of each with a
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

import dataclasses
from typing import Mapping

import sympy
from structlog import get_logger

__all__ = [
    "EPSILON",
    "MAX_HALF_WIDTH",
    "MAX_INLINE_OPS",
    "MAX_SERIES_OPS",
    "MIN_HALF_WIDTH",
    "RemovablePole",
    "agrees_numerically",
    "default_values",
    "denominator_factors",
    "factor_roots",
    "half_width",
    "guard",
    "inline",
    "is_removable",
    "removable_poles",
    "rewrite",
    "taylor",
]

logger = get_logger()

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

#: Narrowest guard window allowed, in the guarded variable's own units.
MIN_HALF_WIDTH = 1e-8

#: Widest guard window allowed, in the guarded variable's own units.
MAX_HALF_WIDTH = 1e-1

#: float64 machine epsilon, the round-off scale the fallback window rule
#: balances truncation against.
EPSILON = 2.220446049250313e-16

#: Give up inlining past this many operations. ToRORd_dyn_chloride's largest
#: candidate reaches 939, so this is loose: it bounds the worst case on a model
#: nobody has tried yet rather than excluding anything in the bundled set.
MAX_INLINE_OPS = 5000

#: Do not series-expand an expression larger than this many operations.
#: ``sympy.series`` has no useful bound of its own: on ToRORd's 267-operation
#: ``E1_i`` it did not finish in 100 s. The largest guard any bundled model
#: needs is well under this.
MAX_SERIES_OPS = 150


def denominator_factors(expr: sympy.Expr) -> tuple[sympy.Expr, ...]:
    """The factors of the outermost denominators in ``expr``.

    Found structurally, as the bases of negative powers, rather than with
    ``sympy.together``: combining an expression into one fraction first was
    most of the cost of the scan on ToRORd.

    Only *outermost* denominators are returned; a denominator's own
    denominators are not looked inside. ``1/(1 + c/(K + x)**2)`` yields
    ``1 + c/(K + x)**2``, not ``K + x``. Looking inside finds real removable
    singularities too -- the formula above is regular at ``x = -K`` even
    though it divides by zero there -- but measured on the bundled models it
    triples the guarded set on ToRORd (8 to 24, mostly concentration buffering
    terms at negative or zero concentrations), and ``sympy.series`` did not
    finish within 170 s on ORdmm_Land. The outermost denominators give exactly
    the GHK and gate-rate poles.

    Parameters
    ----------
    expr : sympy.Expr
        Any expression.

    Returns
    -------
    tuple[sympy.Expr, ...]
        The distinct factors, in a deterministic order. An expression with no
        denominator yields ``(1,)``.
    """
    factors: dict[sympy.Expr, None] = {}
    stack = [expr]
    while stack:
        node = stack.pop()
        if isinstance(node, sympy.Pow) and node.exp.is_negative:
            for factor in sympy.Mul.make_args(node.base):
                if not factor.is_number:
                    factors[factor] = None
            continue
        stack.extend(node.args)
    if not factors:
        return (sympy.S.One,)
    return tuple(sorted(factors, key=sympy.default_sort_key))


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
    return -intercept / slope


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
    # Split by dependence on `var`, not by the presence of `exp`: in
    # `exp(x) - exp(2)` the second term is an exponential, but a constant one.
    exponentials = [term for term in factor.args if var in term.free_symbols]
    constants = [term for term in factor.args if var not in term.free_symbols]
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

    ratio = -intercept / coefficient
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
        # evaluate(False): a default xreplace rebuilds every ancestor with
        # evaluate=True, which folds `exp(-0.04*(V + 23))` into
        # `0.398519041084514*exp(-0.04*V)`. After that the pole is no longer
        # exactly at -23 in any arithmetic, and the series about -23 picks up
        # a Laurent term. `sympytools.rhs_matrix` guards against the same
        # folding for the same reason.
        with sympy.core.parameters.evaluate(False):
            expr = expr.xreplace(substitutions)
        if sympy.count_ops(expr) > MAX_INLINE_OPS:
            return None


_DEPENDENCY_CACHE: list = []


def _depending_on(
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    seeds: set[sympy.Symbol],
) -> set[sympy.Symbol]:
    """Cached front for :func:`_compute_depending_on`.

    Called once per (assignment, variable) pair against the same model, so
    recomputing the fixed point every time dominated the scan. The cache holds
    only the most recent ``definitions`` object and compares by identity, so it
    can never serve a stale answer for a different model.
    """
    key = frozenset(seeds)
    if not _DEPENDENCY_CACHE or _DEPENDENCY_CACHE[0] is not definitions:
        _DEPENDENCY_CACHE[:] = [definitions, {}]
    by_seeds = _DEPENDENCY_CACHE[1]
    if key not in by_seeds:
        by_seeds[key] = frozenset(_compute_depending_on(definitions, set(seeds)))
    return set(by_seeds[key])


def _compute_depending_on(
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
    exponent = _leading_exponent(expr, var, value)
    return exponent is not None and bool(exponent >= 0)


def _leading_exponent(expr: sympy.Expr, var: sympy.Symbol, value: sympy.Expr) -> sympy.Expr | None:
    """Leading exponent of ``expr`` in ``var - value``, or None if sympy fails."""
    offset = sympy.Dummy("offset", positive=True)
    try:
        _, exponent = expr.subs(var, value + offset).leadterm(offset)
    except _SERIES_ERRORS:
        return None
    return exponent


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
    series = sympy.expand(series)
    # A replacement must be a polynomial in `var`. A Laurent term left over
    # from float round-off -- e.g. -5.7e-14/(V + 23.000000000000011) -- is
    # negligible at the window edge, so the numeric check cannot see it, and
    # infinite a hair away from the guarded point.
    if not series.is_polynomial(var):
        return None
    return series


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
    replacement: sympy.Expr | None = None,
) -> float | None:
    r"""Half-width of the window in which the Taylor branch is used.

    Placed at the crossover between the two branches' float64 errors,
    measured on the *derivative* -- the quantity the Rush-Larsen linearized
    block needs. Near the pole the direct formula loses precision to
    cancellation; far from it the truncated series loses it to truncation.
    Where they cross, both are as accurate as either gets.

    The crossover is found *empirically*: on a geometric grid of candidate
    half-widths, the series derivative and the direct derivative are both
    evaluated in float64 at ``value +- delta``, and the window edge goes where
    they agree best. Each is accurate on its own side of the crossover, so
    their disagreement is V-shaped in ``delta`` and bottoms out there.

    An analytic rule -- balancing truncation :math:`k|a_k|\delta^{k-1}/|a_1|`
    against round-off :math:`\epsilon/\delta^2` -- is used only as a fallback
    when the expression cannot be evaluated numerically. It misplaces the
    window whenever the pole's local variable is scaled: Beeler-Reuter's
    ``i_K1`` cancels in ``exp(-0.04*(V + 23)) - 1``, so its round-off grows
    like :math:`\epsilon/(0.04\,\delta)^2` and the analytic rule puts the edge
    at 4.9e-3, where the direct derivative is still 8.2e-9 off. The true
    crossover is near 2e-2, with both branches under 7e-10.

    Calibrating on the *value* rather than the derivative is worse again: on
    ToRORd's ``INab`` it gives 0.1, where the series derivative is 4.9e-10
    off a 50-digit reference and the direct derivative only 3.0e-13 off.

    The result is clamped to ``[MIN_HALF_WIDTH, MAX_HALF_WIDTH]``.

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
        Default values for parameters, states and intermediates. A window
        half-width has to be a number.
    replacement : sympy.Expr | None, optional
        The Taylor replacement, if already computed; computed here otherwise.

    Returns
    -------
    float | None
        The half-width, or None if neither the empirical search nor the
        analytic fallback could be evaluated.
    """
    if replacement is None:
        replacement = taylor(expr, var, value, order)
    if replacement is not None:
        delta = _empirical_half_width(expr, replacement, var, value, defaults)
        if delta is not None:
            return delta
    return _analytic_half_width(expr, var, value, order, defaults)


#: Candidate half-widths for the empirical search: seven per decade across
#: the allowed range.
_HALF_WIDTH_GRID = tuple(
    MIN_HALF_WIDTH * (MAX_HALF_WIDTH / MIN_HALF_WIDTH) ** (i / 49) for i in range(50)
)


def _empirical_half_width(
    expr: sympy.Expr,
    replacement: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    defaults: Mapping[sympy.Symbol, float],
) -> float | None:
    """The grid half-width where the two branches' derivatives agree best."""
    import numpy

    center = _numeric(value, defaults)
    if center is None:
        return None
    others = {s: sympy.Float(x) for s, x in defaults.items() if s != var}
    try:
        direct = sympy.lambdify(var, sympy.diff(expr, var).xreplace(others), "numpy")
        series = sympy.lambdify(var, sympy.diff(replacement, var).xreplace(others), "numpy")
    except _SERIES_ERRORS:
        return None

    def disagreement(delta: float) -> float:
        worst = 0.0
        # Four points rather than two, so that a lucky cancellation in the
        # noisy direct formula at one point cannot fake a good agreement.
        for point in (center - delta, center + delta, center - 1.13 * delta, center + 1.13 * delta):
            with numpy.errstate(all="ignore"):
                try:
                    d = complex(direct(numpy.float64(point)))
                    s = complex(series(numpy.float64(point)))
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    return float("inf")
            if d.imag or s.imag or not (numpy.isfinite(d.real) and numpy.isfinite(s.real)):
                return float("inf")
            scale = max(abs(d.real), abs(s.real), 1e-300)
            worst = max(worst, abs(d.real - s.real) / scale)
        return worst

    scores = [(delta, disagreement(delta)) for delta in _HALF_WIDTH_GRID]
    finite = [score for _, score in scores if score != float("inf")]
    if not finite:
        return None
    best = min(finite)
    # Widest window that is within a factor of two of the best agreement: on
    # a flat curve -- a replacement that is exact, say -- that is the upper
    # clamp rather than an arbitrary point on the plateau.
    return max(delta for delta, score in scores if score <= 2 * best + 1e-300)


def _analytic_half_width(
    expr: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    order: int,
    defaults: Mapping[sympy.Symbol, float],
) -> float | None:
    r"""Fallback half-width from series coefficients alone.

    With :math:`k` the first nonzero index above ``order``, the crossing of
    derivative truncation and round-off is at
    :math:`\delta^{k+1} = \epsilon |a_1| / (k |a_k|)`. Ignores the local
    variable's scale; see :func:`half_width`.
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
    center. ``delta`` is placed where the two branches cross over, so both are
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
    center = _numeric(value, defaults)
    if center is None:
        return False
    for point in (center - delta, center + delta):
        substitution = {var: sympy.Float(point)}
        original = _numeric(expr.subs(substitution), defaults)
        proposed = _numeric(replacement.subs(substitution), defaults)
        if original is None or proposed is None:
            return False
        scale = max(abs(original), abs(proposed), 1e-300)
        if abs(original - proposed) / scale > tolerance:
            return False
    return True


@dataclasses.dataclass(frozen=True)
class RemovablePole:
    """A removable pole and the replacement that covers it.

    Attributes
    ----------
    var : sympy.Symbol
        The state variable the pole lives in.
    value : sympy.Expr
        Where the pole is. Free of state symbols, and real and finite at
        default parameter values, so that the window is a compile-time
        constant.
    replacement : sympy.Expr
        Truncated Taylor series, of order at least 1.
    half_width : float
        Half-width of the window in which ``replacement`` is used.
    rewritten : sympy.Expr
        The original expression, rewritten in ``var``. This is the branch
        taken *outside* the window, and it has to be written in ``var`` too:
        the forward-mode sweep in :mod:`gotranx.linearization` is seeded at
        ``var``, and a branch written in terms of an intermediate would
        contribute nothing to that state's tangent.
    """

    var: sympy.Symbol
    value: sympy.Expr
    replacement: sympy.Expr
    half_width: float
    rewritten: sympy.Expr


def _exact(expr: sympy.Expr) -> sympy.Expr:
    """``expr`` with its float literals turned into exact rationals.

    Pole locations must be found on an exact copy, not on the expression as
    written. Beeler-Reuter's ``i_K1`` has the factor
    ``1 - exp(-0.04*(V + 23))``; in float arithmetic sympy folds that to
    ``1 - 0.398519041084514*exp(-0.04*V)`` and the root solves to
    -23.000000000000004 rather than -23. Expanding a series about that
    location -- 3.6e-15 off the actual pole -- makes ``sympy.series`` return
    plain ``0``, so the replacement is worthless and the guard is dropped.
    Rationalized, the factor is ``1 - exp(-V/25 - 23/25)`` and the root is
    exactly -23.

    This is the same weakness as the ``sympy.limit`` failure that rules
    ``limit`` out of this module: float coefficients defeat sympy's exact
    machinery, and ``.ode`` files are written with float coefficients.
    """
    return sympy.nsimplify(expr, rational=True)


def _underlying_states(
    symbols: set[sympy.Symbol],
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    states: frozenset[sympy.Symbol],
) -> set[sympy.Symbol]:
    """The states that ``symbols`` transitively depend on."""
    found: set[sympy.Symbol] = set()
    seen: set[sympy.Symbol] = set()
    frontier = set(symbols)
    while frontier:
        symbol = frontier.pop()
        if symbol in seen:
            continue
        seen.add(symbol)
        if symbol in states:
            found.add(symbol)
        elif symbol in definitions:
            frontier |= definitions[symbol].free_symbols
    return found


def removable_poles(
    expr: sympy.Expr,
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    states: frozenset[sympy.Symbol],
    defaults: Mapping[sympy.Symbol, float],
    order: int = 3,
) -> tuple[RemovablePole, ...]:
    """Every removable pole of ``expr`` in a state variable.

    Only expressions whose denominator contains a state-dependent symbol are
    examined at all; a denominator that is a parameter or a literal cannot
    vanish for a state-dependent reason. That pre-filter is what keeps the
    scan affordable -- ``dv_dt = (I_stim - ...)/C`` is skipped without
    inlining its 10,013-operation cone, because ``C`` is a parameter.

    A pole is kept only if all of the following hold:

    * its location is free of state symbols, so the window half-width is a
      compile-time constant rather than a moving target;
    * its location is a real, finite number at default parameter values
      (tentusscher_panfilov's ``Ca_i`` buffering roots are
      ``-K_buf_c +- sqrt(-Buf_c*K_buf_c)``, imaginary for positive
      parameters);
    * it is removable, by leading exponent;
    * a truncated series exists; and
    * that series agrees numerically with the original at the window edge.

    Parameters
    ----------
    expr : sympy.Expr
        The assignment's expression, as written in the model.
    definitions : Mapping[sympy.Symbol, sympy.Expr]
        Every intermediate symbol in the model and its defining expression.
    states : frozenset[sympy.Symbol]
        The model's state symbols.
    defaults : Mapping[sympy.Symbol, float]
        Default values for every parameter and state.
    order : int, optional
        Order of the Taylor replacement, by default 3.

    Returns
    -------
    tuple[RemovablePole, ...]
        In a deterministic order: by variable name, then by location, so that
        generated code does not depend on set iteration order.
    """
    state_dependent = set(states) | _depending_on(definitions, set(states))

    denominator_symbols: set[sympy.Symbol] = set()
    for factor in denominator_factors(expr):
        denominator_symbols |= factor.free_symbols & state_dependent
    if not denominator_symbols:
        return ()

    found: list[RemovablePole] = []
    for var in sorted(_underlying_states(denominator_symbols, definitions, states), key=str):
        rewritten = inline(expr, definitions, var)
        if rewritten is None or sympy.count_ops(rewritten) > MAX_SERIES_OPS:
            # Checked here rather than only before expanding: nothing this
            # large can be guarded, so there is no point scanning its factors.
            logger.debug("Expression too large to guard", var=str(var))
            continue
        # Cheap pass on the float form: most candidates have no root in a
        # matched shape at all, and rationalizing is the expensive step.
        if not any(factor_roots(factor, var) for factor in denominator_factors(rewritten)):
            continue
        exact = _exact(rewritten)

        seen: set[sympy.Expr] = set()
        for factor in denominator_factors(exact):
            for value in factor_roots(factor, var):
                if value.free_symbols & states:
                    logger.debug(
                        "Skipping a pole whose location depends on a state",
                        var=str(var),
                        value=str(value),
                    )
                    continue
                if _numeric(value, defaults) is None:
                    logger.debug(
                        "Skipping a pole whose location is not a real number",
                        var=str(var),
                        value=str(value),
                    )
                    continue
                if value in seen:
                    continue
                seen.add(value)

                pole = _pole_at(rewritten, exact, var, value, order, defaults)
                if pole is not None:
                    found.append(pole)
    return tuple(sorted(found, key=lambda pole: (str(pole.var), str(pole.value))))


def _pole_at(
    rewritten: sympy.Expr,
    exact: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    order: int,
    defaults: Mapping[sympy.Symbol, float],
) -> RemovablePole | None:
    """Classify and, if removable, build the guard for one candidate pole.

    Tried on the expression as written first, and on the exact copy only if
    that fails. The float form is much the cheaper of the two -- the exact
    series for Beeler-Reuter's ``i_K1`` runs to hundreds of terms before
    ``nfloat`` collapses it to the same four coefficients -- so it is worth
    preferring, with the exact form as the fallback for the cases where float
    arithmetic defeats the series machinery.
    """
    if sympy.count_ops(rewritten) > MAX_SERIES_OPS:
        logger.debug("Expression too large to expand", var=str(var), value=str(value))
        return None
    if _looks_like_a_genuine_pole(exact, var, value, defaults):
        return None

    # The exact form is consulted only if sympy *fails* on the float form, not
    # if the float form says the pole is genuine: genuine poles are most of the
    # candidates, and a second leadterm on the exact form roughly doubled the
    # scan for no change in the verdict.
    exponent = _leading_exponent(rewritten, var, value)
    if exponent is None:
        exponent = _leading_exponent(exact, var, value)
    if exponent is None or not bool(exponent >= 0):
        return None

    # The series is taken on the float form, about the *exact* location. The
    # exact form's series is no alternative: its coefficients are sums of
    # rational multiples of exp(rational) that nfloat cannot collapse without
    # losing every digit to cancellation.
    replacement = taylor(rewritten, var, value, order)
    if replacement is not None:
        replacement = sympy.nfloat(replacement, n=17)
        delta = half_width(rewritten, var, value, order, defaults, replacement)
        if delta is not None and agrees_numerically(
            rewritten, replacement, var, value, delta, defaults
        ):
            return RemovablePole(
                var=var,
                value=sympy.nfloat(value, n=17),
                replacement=replacement,
                half_width=delta,
                rewritten=rewritten,
            )

    logger.warning(
        "Dropping a guard whose replacement failed its numeric check",
        var=str(var),
        value=str(value),
    )
    return None


def _looks_like_a_genuine_pole(
    exact: sympy.Expr,
    var: sympy.Symbol,
    value: sympy.Expr,
    defaults: Mapping[sympy.Symbol, float],
) -> bool:
    """Cheap numeric screen that rejects genuine poles before ``leadterm``.

    Most candidates are genuine poles, and proving that symbolically -- one
    ``leadterm`` each, roughly 90 ms on ToRORd -- was most of the scan. A
    genuine pole of order ``k`` grows by ``1e8**k`` between offsets of 1e-12
    and 1e-20; a removable singularity does not grow at all. Evaluated on the
    exact form in 60-digit arithmetic, so the 1e-20 offset is far above
    round-off.

    Only ever *rejects*. A candidate that passes, or that cannot be evaluated
    here, still goes through the symbolic leading-exponent test.
    """
    import mpmath

    others = {symbol: sympy.Rational(repr(x)) for symbol, x in defaults.items() if symbol != var}
    try:
        center = sympy.sympify(value).xreplace(others)
        function = sympy.lambdify(var, exact.xreplace(others), "mpmath")
        with mpmath.workdps(60):
            origin = mpmath.mpf(sympy.N(center, 60))
            near = abs(function(origin + mpmath.mpf("1e-20")))
            far = abs(function(origin + mpmath.mpf("1e-12")))
    except Exception:  # noqa: BLE001 -- any failure just defers to leadterm
        return False
    if not (mpmath.isfinite(near) and mpmath.isfinite(far)) or far == 0:
        return False
    return bool(near > 1e4 * far)


def guard(expr: sympy.Expr, poles: tuple[RemovablePole, ...]) -> sympy.Expr:
    """Wrap ``expr`` in one ``Piecewise`` per pole.

    Parameters
    ----------
    expr : sympy.Expr
        The expression to guard.
    poles : tuple[RemovablePole, ...]
        The poles to cover. May be empty, in which case ``expr`` is returned
        unchanged.

    Returns
    -------
    sympy.Expr
        ``Piecewise((replacement, Abs(var - value) < half_width), (expr, True))``,
        nested when there is more than one pole. The innermost fallback is the
        last pole's ``rewritten`` form rather than ``expr`` itself, so the
        guarded expression is written in a variable the AD sweep is seeded at.
        Every ``rewritten`` form is mathematically equal to ``expr``, so which
        one ends up innermost does not change the value.
    """
    if not poles:
        return expr
    guarded = poles[-1].rewritten
    for pole in poles:
        guarded = sympy.Piecewise(
            (
                pole.replacement,
                sympy.Abs(pole.var - pole.value) < sympy.Float(pole.half_width),
            ),
            (guarded, True),
        )
    return guarded


def rewrite(
    expr: sympy.Expr,
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    states: frozenset[sympy.Symbol],
    defaults: Mapping[sympy.Symbol, float],
    order: int = 3,
) -> sympy.Expr:
    """Guard every removable pole of ``expr``, or return it unchanged.

    This is the one entry point the model layer needs.

    Parameters
    ----------
    expr : sympy.Expr
        The assignment's expression.
    definitions : Mapping[sympy.Symbol, sympy.Expr]
        Every intermediate symbol in the model and its defining expression.
    states : frozenset[sympy.Symbol]
        The model's state symbols.
    defaults : Mapping[sympy.Symbol, float]
        Default values for every parameter and state.
    order : int, optional
        Order of the Taylor replacement, by default 3.

    Returns
    -------
    sympy.Expr
        The guarded expression, or ``expr`` itself -- the identical object, so
        callers can test with ``is`` -- if it has no removable pole.
    """
    poles = removable_poles(expr, definitions, states, defaults, order=order)
    if not poles:
        return expr
    logger.debug(
        "Guarding removable poles",
        poles=[(str(p.var), str(p.value), p.half_width) for p in poles],
    )
    return guard(expr, poles)


def default_values(
    definitions: Mapping[sympy.Symbol, sympy.Expr],
    base: Mapping[sympy.Symbol, float],
) -> dict[sympy.Symbol, float]:
    """Default values for every intermediate that can be evaluated.

    :func:`inline` leaves intermediates that do not depend on the pole
    variable as opaque symbols -- ToRORd's ``ICab`` keeps ``gamma_cai`` and
    ``gamma_cao`` after being rewritten in ``v``. The window half-width and the
    numeric check both need numbers, so those symbols need defaults too;
    without them every GHK guard in ToRORd is silently dropped.

    Parameters
    ----------
    definitions : Mapping[sympy.Symbol, sympy.Expr]
        Every intermediate and its defining expression, in dependency order
        (as :meth:`gotranx.ode.ODE.sorted_assignments` yields them).
    base : Mapping[sympy.Symbol, float]
        Default values for parameters and states.

    Returns
    -------
    dict[sympy.Symbol, float]
        ``base`` plus every intermediate whose value is a real, finite number
        at those defaults. An intermediate that is not -- ToRORd's bundled
        state has ``CaTrpn = 0``, and a guard comparing ``CaTrpn**(-ntm/2)``
        against 100 raises under exact arithmetic -- is simply left out, and
        any pole whose checks need it is left unguarded.
    """
    values = dict(base)
    for symbol, definition in definitions.items():
        if not definition.free_symbols <= set(values):
            continue
        try:
            value = _numeric(definition, values)
        except (TypeError, ValueError, ZeroDivisionError):
            value = None
        if value is not None:
            values[symbol] = value
    return values
