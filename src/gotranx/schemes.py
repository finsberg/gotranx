from __future__ import annotations
import typing
import sympy
from structlog import get_logger

from . import atoms
from .ode import ODE
from . import sympytools

logger = get_logger()


class Scheme(typing.Protocol):
    def __call__(self, ode: ODE, dt: sympy.Symbol, name: str = "values") -> list[sympy.Eq]:
        ...


def valid_schmemes() -> list[str]:
    return [
        "forward_explicit_euler",
        "forward_generalized_rush_larsen",
    ]


def get_scheme(scheme: str) -> Scheme:
    """Get the scheme function from a string"""
    if scheme in ["forward_euler", "forward_explicit_euler", "euler", "explicit_euler"]:
        return forward_explicit_euler
    elif scheme in ["forward_generalized_rush_larsen", "generalized_rush_larsen"]:
        return forward_generalized_rush_larsen
    else:
        raise ValueError(f"Unknown scheme {scheme}")


def fraction_numerator_is_nonzero(expr):
    """Perform a very cheap check to detect if a fraction is definitely non-zero."""

    if isinstance(expr, sympy.Pow):
        # check if the expression is on the form a**-1
        a, b = expr.args
        if b is sympy.S.NegativeOne:
            return True
        else:
            # we won't do any further checks
            return False
    elif isinstance(expr, sympy.Mul):
        # check if all factors are non-zero
        args = expr.args
        certainly_nonzero_args = []
        potentially_nonzero_args = []
        for e in args:
            if len(e.free_symbols) == 0 and e.is_nonzero:
                certainly_nonzero_args.append(e)
            else:
                potentially_nonzero_args.append(e)

        if len(potentially_nonzero_args) == 0:
            # all factors are certainly nonzero
            return True

        # check all potentially non-zero factors
        for e in potentially_nonzero_args:
            if not fraction_numerator_is_nonzero(e):
                return False
        else:
            return True
    else:
        return False


def forward_explicit_euler(ode: ODE, dt: sympy.Symbol, name: str = "values") -> list[sympy.Eq]:
    """Generate forward Euler equations for the ODE"""
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    i = 0
    for x in ode.sorted_assignments():
        eqs.append(sympy.Eq(x.symbol, x.expr))
        if isinstance(x, atoms.StateDerivative):
            eqs.append(
                sympy.Eq(
                    values[i],
                    x.state.symbol + dt * x.symbol,
                )
            )
            i += 1

    return eqs


def forward_generalized_rush_larsen(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    delta=1e-8,
) -> list[sympy.Eq]:
    """Generate forward Generalized Rush Larsen equations for the ODE"""
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    i = 0
    for x in ode.sorted_assignments():
        eqs.append(sympy.Eq(x.symbol, x.expr))

        if not isinstance(x, atoms.StateDerivative):
            continue

        expr_diff = x.expr.diff(x.state.symbol)

        if expr_diff.is_zero:
            # Use forward Euler
            eqs.append(
                sympy.Eq(
                    values[i],
                    x.state.symbol + dt * x.symbol,
                )
            )
            i += 1
            continue

        linearized_name = x.name + "_linearized"
        linearized = sympy.Symbol(linearized_name)
        eqs.append(sympy.Eq(linearized, expr_diff))

        need_zero_div_check = not fraction_numerator_is_nonzero(expr_diff)
        if not need_zero_div_check:
            logger.debug(f"{linearized_name} cannot be zero. Skipping zero division check")

        RL_term = x.symbol / linearized * (sympy.exp(linearized * dt) - 1)
        if need_zero_div_check:
            RL_term = sympytools.Conditional(
                abs(linearized) > delta,
                RL_term,
                dt * x.symbol,
            )
        eqs.append(
            sympy.Eq(
                values[i],
                x.state.symbol + RL_term,
            )
        )
        i += 1
    return eqs