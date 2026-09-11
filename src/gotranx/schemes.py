from __future__ import annotations
import typing
from types import CodeType

import sympy
from structlog import get_logger

from . import atoms
from .ode import ODE
from .linearization import diagonal_jacobian
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


def _taken_names(ode: ODE) -> set[str]:
    """Every identifier already in use in the ODE.

    Used to keep generated CSE temporaries from shadowing a model symbol. The
    `.ode` grammar uses Lark's CNAME, which permits a leading underscore, so a
    `_` prefix is not by itself collision-proof.
    """
    names = {s.name for s in ode.states} | {p.name for p in ode.parameters}
    names |= {a.name for a in ode.sorted_assignments()}
    return names


def _linearized_assignments(
    derivative_name: str,
    expr: sympy.Expr,
    taken: set[str],
) -> tuple[list[tuple[sympy.Symbol, sympy.Expr]], sympy.Expr]:
    """Factor shared subexpressions out of one linearized expression.

    CSE is applied per state rather than jointly across all states. Joint CSE
    gives a lower operation count but leaves every temporary live across the
    whole function, which on the vectorized backends means one live array per
    temporary. Per-state temporaries are dead by the end of the state's own
    update.

    Returns the temporaries, in the order they must be emitted, and the reduced
    expression to assign to ``<derivative_name>_linearized``.
    """

    def fresh():
        i = 0
        while True:
            name = f"_{derivative_name}_linearized_{i}"
            if name not in taken:
                yield sympy.Symbol(name, real=True)
            i += 1

    replacements, reduced = sympy.cse([expr], symbols=fresh(), optimizations="basic")
    return replacements, reduced[0]


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

    The linearization :math:`g = \partial f_i / \partial y_i` is the diagonal of
    the Jacobian of the full right-hand side, computed by differentiating through
    intermediate expressions. It therefore does not depend on whether the model
    author named a subexpression.

    If :math:`g` is zero the update reduces to :math:`x_{n+1} = x_n + dt f`, which
    is the exact :math:`g \to 0` limit of the expression above, not a fallback.

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
        States to integrate with the Rush-Larsen update; all others use forward
        Euler. By default None, which makes this scheme equivalent to explicit
        Euler. Every state has a usable linearization, so this is a cost/accuracy
        trade-off rather than a statement about which states can be linearized:
        the Rush-Larsen update costs an exponential per state per step.

    Returns
    -------
    list[str]
        A list of equations as strings

    """
    if stiff_states is None:
        stiff_states = []
    logger.debug("Generating hybrid Rush-Larsen scheme", stiff_states=stiff_states)
    stiff_states_set = set(stiff_states)
    found_stiff_states_set = set()
    not_linearizable = set()
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    taken = _taken_names(ode)
    i = 0
    for x in ode.sorted_assignments(remove_unused=remove_unused):
        eqs.append(printer(x.symbol, x.expr, use_variable_prefix=True))

        if not isinstance(x, atoms.StateDerivative):
            continue

        expr_diff = jacobian[x.state.name]
        state_is_stiff = x.state.name in stiff_states_set

        # Record the state as found before deciding how to integrate it, so a
        # state that is present but cannot be linearized is never reported as
        # missing from the ODE.
        if state_is_stiff:
            found_stiff_states_set.add(x.state.name)
            if expr_diff.is_zero:
                not_linearizable.add(x.state.name)

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

        logger.debug(f"State {x.state.name} is stiff")
        replacements, expr_diff = _linearized_assignments(x.name, expr_diff, taken)
        for symbol, sub_expr in replacements:
            eqs.append(printer(symbol, sub_expr, use_variable_prefix=True))

        linearized_name = x.name + "_linearized"
        linearized = sympy.Symbol(linearized_name)
        eqs.append(printer(linearized, expr_diff, use_variable_prefix=True))

        # `expr_diff` is the post-CSE reduced expression, so this check runs on
        # the CSE'd form: if a provably-nonzero factor is hidden behind an
        # opaque temporary, the check can't see through it and conservatively
        # asks for a zero-division guard that a pre-CSE check would have
        # skipped. That's an accepted cost of factoring first, not a bug.
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

    if not_linearizable:
        logger.warning(
            "The following states were marked as stiff but their derivative does "
            "not depend on the state, so they use forward Euler: "
            f"{sorted(not_linearizable)}"
        )
    missing = stiff_states_set.difference(found_stiff_states_set)
    if missing:
        logger.warning(
            f"The following states were marked as stiff but not found in the ODE: {sorted(missing)}"
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

    The linearization :math:`g = \partial f_i / \partial y_i` is the diagonal of
    the Jacobian of the full right-hand side, computed by differentiating through
    intermediate expressions. It therefore does not depend on whether the model
    author named a subexpression.

    If :math:`g` is zero the update reduces to :math:`x_{n+1} = x_n + dt f`, which
    is the exact :math:`g \to 0` limit of the expression above, not a fallback.

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
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    taken = _taken_names(ode)
    i = 0
    for x in ode.sorted_assignments(remove_unused=remove_unused):
        eqs.append(printer(x.symbol, x.expr, use_variable_prefix=True))

        if not isinstance(x, atoms.StateDerivative):
            continue

        expr_diff = jacobian[x.state.name]

        if expr_diff.is_zero:
            # df/dx is genuinely zero, so f*dt is the exact dx -> 0 limit of the
            # exponential update, not a fallback.
            logger.debug(
                f"d{x.state.name}/dt does not depend on {x.state.name}; "
                "the exponential update reduces to forward Euler"
            )
            eqs.append(
                printer(
                    values[i],
                    x.state.symbol + dt * x.symbol,
                )
            )
            i += 1
            continue

        replacements, expr_diff = _linearized_assignments(x.name, expr_diff, taken)
        for symbol, sub_expr in replacements:
            eqs.append(printer(symbol, sub_expr, use_variable_prefix=True))

        linearized_name = x.name + "_linearized"
        linearized = sympy.Symbol(linearized_name)
        eqs.append(printer(linearized, expr_diff, use_variable_prefix=True))

        # `expr_diff` is the post-CSE reduced expression, so this check runs on
        # the CSE'd form: if a provably-nonzero factor is hidden behind an
        # opaque temporary, the check can't see through it and conservatively
        # asks for a zero-division guard that a pre-CSE check would have
        # skipped. That's an accepted cost of factoring first, not a bug.
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
