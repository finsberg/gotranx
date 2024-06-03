from __future__ import annotations
import typing
import sympy
from enum import Enum

from structlog import get_logger

from . import atoms
from .ode import ODE
from . import sympytools

logger = get_logger()


def default_printer(
    lhs: sympy.Symbol | sympy.IndexedBase,
    rhs: sympy.Expr,
    use_variable_prefix: bool = False,
) -> str:
    from sympy.codegen.ast import Assignment
    from sympy.printing import pycode

    return pycode(Assignment(lhs, rhs))


class printer_func(typing.Protocol):
    def __call__(
        self,
        lhs: sympy.Symbol | sympy.IndexedBase,
        rhs: sympy.Expr,
        use_variable_prefix: bool = False,
    ) -> str: ...


class scheme_func(typing.Protocol):
    def __call__(
        self,
        ode: ODE,
        dt: sympy.Symbol,
        name: str = "values",
        printer: printer_func = default_printer,
        remove_unused: bool = False,
    ) -> list[str]: ...


class Scheme(str, Enum):
    forward_explicit_euler = "forward_explicit_euler"
    forward_generalized_rush_larsen = "forward_generalized_rush_larsen"


def get_scheme(scheme: str) -> scheme_func:
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


def forward_explicit_euler(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
) -> list[str]:
    """Generate forward Euler equations for the ODE

    The forward Euler scheme is given by

    $$
    x_{n+1} = x_n + dt f(x_n, t_n)
    $$

    Parameters
    ----------
    ode : ODE
        The ODE
    dt : sympy.Symbol
        The time step
    name : str, optional
        Name of array to be returned by the scheme, by default "values"
    printer : printer_func, optional
        A code printer, by default default_printer
    remove_unused : bool, optional
        Remove unused variables, by default False

    Returns
    -------
    list[str]
        A list of equations as strings

    """
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    i = 0
    for x in ode.sorted_assignments(remove_unused=remove_unused):
        eqs.append(printer(x.symbol, x.expr, use_variable_prefix=True))
        if isinstance(x, atoms.StateDerivative):
            eqs.append(
                printer(
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
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
) -> list[str]:
    """Generate the forward generalized Rush-Larsen scheme for the ODE

    The forward generalized Rush-Larsen scheme is given by

    $$
    x_{n+1} = x_n + \\frac{f(x_n, t_n)}{g(x_n, t_n)} \\left( e^{g(x_n, t_n) dt} - 1 \\right)
    $$

    where $g(x_n, t_n)$ is the linearization of $f(x_n, t_n)$ around $x_n$

    We fall back to forward Euler if the derivative is zero.

    Parameters
    ----------
    ode : ODE
        The ODE
    dt : sympy.Symbol
        The time step
    name : str, optional
        Name of array to be returned by the scheme, by default "values"
    printer : printer_func, optional
        A code printer, by default default_printer
    remove_unused : bool, optional
        Remove unused variables, by default False
    delta : float, optional
        Tolerance for zero division check, by default 1e-8

    Returns
    -------
    list[str]
        A list of equations as strings
    """
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    i = 0
    for x in ode.sorted_assignments(remove_unused=remove_unused):
        eqs.append(printer(x.symbol, x.expr, use_variable_prefix=True))

        if not isinstance(x, atoms.StateDerivative):
            continue

        expr_diff = x.expr.diff(x.state.symbol)

        if expr_diff.is_zero:
            # Use forward Euler
            eqs.append(
                printer(
                    values[i],
                    x.state.symbol + dt * x.symbol,
                )
            )
            i += 1
            continue

        linearized_name = x.name + "_linearized"
        linearized = sympy.Symbol(linearized_name)
        eqs.append(printer(linearized, expr_diff, use_variable_prefix=True))

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
            printer(
                values[i],
                x.state.symbol + RL_term,
            )
        )
        i += 1
    return eqs
