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

import sympy

__all__ = [
    "denominator_factors",
    "factor_roots",
]


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
