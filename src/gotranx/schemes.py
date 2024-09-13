from __future__ import annotations
import typing
from types import CodeType

import sympy
from structlog import get_logger

from . import atoms
from .ode import ODE
from . import sympytools
from enum import Enum


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
    __code__: CodeType

    def __call__(
        self,
        ode: ODE,
        dt: sympy.Symbol,
        name: str = "values",
        printer: printer_func = default_printer,
        remove_unused: bool = False,
    ) -> list[str]: ...


class Scheme(str, Enum):
    explicit_euler = "explicit_euler"
    generalized_rush_larsen = "generalized_rush_larsen"
    forward_explicit_euler = "forward_explicit_euler"
    forward_generalized_rush_larsen = "forward_generalized_rush_larsen"
    hybrid_rush_larsen = "hybrid_rush_larsen"


def get_scheme(scheme: str) -> scheme_func:
    """Get the scheme function from a string"""
    if scheme in ["forward_euler", "forward_explicit_euler", "euler", "explicit_euler"]:
        func = explicit_euler
    elif scheme in ["forward_generalized_rush_larsen", "generalized_rush_larsen"]:
        func = generalized_rush_larsen
    elif scheme in ["forward_rush_larsen", "rush_larsen", "hybrid_rush_larsen"]:
        func = hybrid_rush_larsen
    else:
        raise ValueError(f"Unknown scheme {scheme}")

    if scheme.startswith("forward_"):
        import warnings

        warnings.warn(
            "member %r is deprecated; %s" % (scheme, "Use the scheme without the forward prefix"),
            DeprecationWarning,
            stacklevel=3,
        )

    # Replace the name of the function
    func.__code__ = func.__code__.replace(co_name=scheme)
    return func


def list_schemes() -> list[str]:
    """List available schemes"""
    return [s.value for s in Scheme]


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


def explicit_euler(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
) -> list[str]:
    r"""Generate forward Euler equations for the ODE

    The forward Euler scheme is given by

    .. math::
        x_{n+1} = x_n + dt f(x_n, t_n)


    Parameters
    ----------
    ode : gotranx.ode.ODE
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
    logger.debug("Generating explicit Euler scheme")
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


def hybrid_rush_larsen(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
) -> list[str]:
    r"""Generate the hybrid Rush-Larsen scheme for the ODE

    The hybrid Rush-Larsen scheme follows the standard Rush_Larsen scheme is given by

    .. math::
        x_{n+1} = x_n + \frac{f(x_n, t_n)}{g(x_n, t_n)} \left( e^{g(x_n, t_n) dt} - 1 \right)

    where :math:`g(x_n, t_n)` is the linearization of :math:`f(x_n, t_n)`
    around :math:`x_n`. The difference between the hybrid and the standard
    is that the user can specify which states are stiff, and the RL scheme
    will only be used for these states. If the derivative
    of a state is zero, the scheme falls back to forward Euler.

    We fall back to forward Euler if the derivative is zero.

    Parameters
    ----------
    ode : gotranx.ODE
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
    stiff_states : list[str] | None, optional
        Stiff states, by default None. If no stiff states are provided,
        the hybrid rush larsen scheme will be the same as the explicit Euler scheme

    Returns
    -------
    list[str]
        A list of equations as strings

    """
    if stiff_states is None:
        stiff_states = []
    logger.debug("Generating hybrid Rush-Larsen scheme", stiff_states=stiff_states)
    stiff_states_set = set(stiff_states) or set()
    found_stiff_states_set = set()
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    i = 0
    for x in ode.sorted_assignments(remove_unused=remove_unused):
        eqs.append(printer(x.symbol, x.expr, use_variable_prefix=True))

        if not isinstance(x, atoms.StateDerivative):
            continue

        expr_diff = x.expr.diff(x.state.symbol)
        state_is_stiff = x.state.name in stiff_states_set

        if not state_is_stiff or expr_diff.is_zero:
            # Use forward Euler
            eqs.append(
                printer(
                    values[i],
                    x.state.symbol + dt * x.symbol,
                )
            )
            i += 1
            continue

        found_stiff_states_set.add(x.state.name)
        logger.debug(f"State {x.state.name} is stiff")
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
    logger.debug(
        "The following states where marked as stiff but not found in the ODE:",
        extra=stiff_states_set.difference(found_stiff_states_set),
    )
    return eqs


def generalized_rush_larsen(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
) -> list[str]:
    r"""Generate the forward generalized Rush-Larsen scheme for the ODE

    The forward generalized Rush-Larsen scheme is given by

    .. math::
        x_{n+1} = x_n + \frac{f(x_n, t_n)}{g(x_n, t_n)} \left( e^{g(x_n, t_n) dt} - 1 \right)


    where :math:`g(x_n, t_n)` is the linearization of :math:`f(x_n, t_n)` around :math:`x_n`

    We fall back to forward Euler if the derivative is zero.

    Parameters
    ----------
    ode : gotranx.ode.ODE
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
    logger.debug("Generating generalized Rush-Larsen scheme")
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
