from __future__ import annotations
import typing
import sympy

if typing.TYPE_CHECKING:  # pragma: no cover
    from .ode import ODE


def states_matrix(ode: "ODE") -> sympy.Matrix:
    return sympy.Matrix([state.symbol for state in ode.sorted_states()])


def rhs_matrix(ode: "ODE", max_tries: int = 20) -> sympy.Matrix:
    intermediates = {x.symbol: x.expr for x in ode.intermediates}
    rhs = sympy.Matrix([state.expr for state in ode.sorted_state_derivatives()])

    num_tries = 0
    while (any([rhs.has(k) for k in intermediates.keys()])) and num_tries < max_tries:
        rhs = rhs.xreplace(intermediates)
        num_tries += 1

    if num_tries == max_tries:
        raise RuntimeError("Maximum number of tries used")
    return rhs


def jacobi_matrix(ode: "ODE") -> sympy.Matrix:
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
            "Cond %s is of type %s, but must be a Relational" " or Boolean." % (cond, type(cond)),
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
    # FIXME: in latest sympy making comparision difficult
    if "<" not in cond.rel_op and ">" not in cond.rel_op:
        TypeError(
            "Expected a lesser or greater than relational for " "a continuous conditional .",
        )

    # Create Heaviside
    H = 1 / (1 + sympy.exp((cond.args[0] - cond.args[1]) / sigma))

    # Desides which should be weighted with 1 and 0
    if ">" in cond.rel_op:
        return true_value * (1 - H) + false_value * H

    return true_value * H + false_value * (1 - H)
