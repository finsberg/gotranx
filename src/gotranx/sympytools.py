from __future__ import annotations
import sympy


def states_matrix(ode) -> sympy.Matrix:
    """Return a sympy matrix of the states in the ODE

    Parameters
    ----------
    ode : gotranx.ode.ODE
        The ODE

    Returns
    -------
    sympy.Matrix
        A sympy matrix of the states in the ODE
    """
    return sympy.Matrix([state.symbol for state in ode.sorted_states()])


def rhs_matrix(ode, max_tries: int = 20) -> sympy.Matrix:
    """Return a sympy matrix of the right hand side of the ODE

    Parameters
    ----------
    ode : gotranx.ode.ODE
        The ODE
    max_tries : int, optional
        Maximum number of tries to try to replace the symbols, by default 20

    Returns
    -------
    sympy.Matrix
        A sympy matrix of the right hand side of the ODE

    Raises
    ------
    RuntimeError
        If the maximum number of tries is reached
    """
    intermediates = {x.symbol: x.expr for x in ode.intermediates}
    rhs = sympy.Matrix([state.expr for state in ode.sorted_state_derivatives()])

    num_tries = 0
    while (any([rhs.has(k) for k in intermediates.keys()])) and num_tries < max_tries:
        # xreplace() rebuilds every ancestor of a substituted symbol using the
        # default (evaluate=True) constructor, which auto-distributes numeric
        # coefficients over sums - e.g. (x - 4.823)/51.12 would silently
        # become 0.0195618153364632*x - 0.0943466353677621 if an intermediate
        # substituted here happens to sit inside such a division. Matches the
        # same fix applied in myokit.gotran_to_myokit.
        with sympy.core.parameters.evaluate(False):
            rhs = rhs.xreplace(intermediates)
        num_tries += 1

    if num_tries == max_tries:
        raise RuntimeError("Maximum number of tries used")
    return rhs


def jacobi_matrix(ode) -> sympy.Matrix:
    """Return the Jacobian matrix of the ODE

    Parameters
    ----------
    ode : gotranx.ode.ODE
        The ODE

    Returns
    -------
    sympy.Matrix
        The Jacobian matrix of the ODE
    """
    return rhs_matrix(ode).jacobian(states_matrix(ode))


def Conditional(cond, true_value, false_value):
    """
    Declares a conditional

    Arguments
    ---------
    cond : A conditional
        The conditional which should be evaluated
    true_value : Any model expression
        Model expression for a true evaluation of the conditional
    false_value : Any model expression
        Model expression for a false evaluation of the conditional
    """
    cond = sympy.sympify(cond)

    from sympy.core.relational import Relational
    from sympy.logic.boolalg import Boolean, BooleanFalse, BooleanTrue

    # If the conditional is a bool it is already evaluated
    if isinstance(cond, (BooleanFalse, BooleanTrue)):
        return true_value if cond else false_value

    if not isinstance(cond, (Relational, Boolean)):
        raise TypeError(
            "Cond %s is of type %s, but must be a Relational or Boolean." % (cond, type(cond)),
        )

    return sympy.functions.Piecewise(
        (true_value, cond),
        (false_value, sympy.sympify(True)),
        evaluate=True,
    )


def ContinuousConditional(cond, true_value, false_value, sigma=1.0):
    """
    Declares a continuous conditional. Instead of a either or result the
    true and false values are weighted with a sigmoidal function which
    either evaluates to 0 or 1 instead of the true or false.

    Arguments
    ---------
    cond : An InEquality conditional
        An InEquality conditional which should be evaluated
    true_value : Any model expression
        Model expression for a true evaluation of the conditional
    false_value : Any model expression
        Model expression for a false evaluation of the conditional
    sigma : float (optional)
        Determines the sharpness of the sigmoidal function
    """

    cond = sympy.sympify(cond)
    # FIXME: Use the rel_op for check, as some changes has been applied
    # FIXME: in latest sympy making comparison difficult
    if "<" not in cond.rel_op and ">" not in cond.rel_op:
        TypeError(
            "Expected a lesser or greater than relational for a continuous conditional .",
        )

    # Create Heaviside
    H = 1 / (1 + sympy.exp((cond.args[0] - cond.args[1]) / sigma))

    # Decides which should be weighted with 1 and 0
    if ">" in cond.rel_op:
        return true_value * (1 - H) + false_value * H

    return true_value * H + false_value * (1 - H)


def cse_hiding_piecewise(exprs, **kwargs):
    """``sympy.cse``, with ``Piecewise`` subtrees held opaque.

    ``sympy.cse`` happily hoists a subexpression out of a ``Piecewise`` branch
    and computes it unconditionally. That is harmless for a branch chosen for
    convenience and fatal for one chosen to avoid a singularity: the whole
    point of ``Piecewise((taylor, Abs(V - v0) < delta), (expr, True))`` is that
    ``expr`` is never evaluated near ``v0``, and a hoisted temporary evaluates
    it everywhere. The same hoist breaks model conditionals that were already
    written defensively -- given two guards sharing
    ``(V + 40)/(exp(-(V + 40)/10) - 1)``, ``cse`` emits it as a temporary that
    is nan at exactly ``V = -40``.

    Every ``Piecewise`` is replaced by a fresh ``Dummy`` before the call, so
    nothing inside one can be factored out. Afterwards, a ``Piecewise`` that
    ended up used more than once becomes a temporary of its own -- ``cse``
    will not do that for us, because by then it is looking at a bare symbol
    and sees nothing worth naming -- and one used exactly once is restored
    inline. So a guard shared between two expressions is still computed once,
    which is most of what CSE was buying here.

    ``hide`` does not descend into a ``Piecewise``, so a nested guard is
    always part of its enclosing guard's subtree and never a temporary of its
    own. That is what makes it safe to emit every guard temporary ahead of
    ``cse``'s own: a guard's expression references only model symbols, never
    another temporary.

    Note that under numpy both branches of the emitted ``where`` are still
    evaluated. That is ``where`` semantics, not a hoist; what this function
    guarantees is that no temporary computed *outside* a guard divides by zero
    at the guarded value.

    Parameters
    ----------
    exprs : list[sympy.Expr]
        The expressions to factor.
    **kwargs
        Passed straight through to ``sympy.cse``.

    Returns
    -------
    tuple[list, list]
        Exactly what ``sympy.cse`` returns: the replacement pairs and the
        reduced expressions.
    """
    holes: dict[sympy.Expr, sympy.Dummy] = {}

    def hide(expr):
        if isinstance(expr, sympy.Piecewise):
            return holes.setdefault(expr, sympy.Dummy(f"_guard_{len(holes)}"))
        if not expr.args:
            return expr
        return expr.func(*[hide(arg) for arg in expr.args])

    hidden = [hide(expr) for expr in exprs]
    replacements, reduced = sympy.cse(hidden, **kwargs)

    uses: dict[sympy.Dummy, int] = {dummy: 0 for dummy in holes.values()}
    for expr in [sub_expr for _, sub_expr in replacements] + list(reduced):
        for dummy in expr.free_symbols:
            if dummy in uses:
                uses[dummy] += 1

    names = kwargs.get("symbols") or sympy.numbered_symbols(prefix="_guard")
    guard_temporaries = []
    restore = {}
    for expr, dummy in holes.items():
        if uses[dummy] > 1:
            symbol = next(names)
            guard_temporaries.append((symbol, expr))
            restore[dummy] = symbol
        else:
            restore[dummy] = expr

    return (
        guard_temporaries
        + [(symbol, sub_expr.xreplace(restore)) for symbol, sub_expr in replacements],
        [expr.xreplace(restore) for expr in reduced],
    )
