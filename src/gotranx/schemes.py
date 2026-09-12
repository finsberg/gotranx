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


class CSEStrategy(str, Enum):
    """How to factor shared subexpressions out of the linearized (diagonal
    Jacobian) expressions the Rush-Larsen schemes emit.

    joint
        One `sympy.cse` across every state's linearized expression at once,
        so a subexpression shared *between* states is computed once. The
        default: measured (on ToRORd) at 1.08x the rhs operation count,
        against per_state's 1.35x, for essentially the same number of live
        temporaries -- strictly better on every backend, since none of them
        (C, Julia, JAX, or CPython itself) frees a temporary before the
        function returns anyway.
    per_state
        One `sympy.cse` per state, as gotranx did before joint was added.
        Every temporary is only ever referenced within that one state's
        update, so this is the strategy to prefer again if the Python
        backend starts emitting `del` for a temporary once it is dead.
    none
        No CSE at all: each `d<state>_dt_linearized` is one fully inlined
        expression. The most operations (9.43x the rhs on ToRORd) but the
        fewest named locals, which is what the vectorized numpy backend
        pays for in live arrays.
    """

    joint = "joint"
    per_state = "per_state"
    none = "none"


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
    names |= set(ode.missing_variables)
    return names


def _linearized_assignments(
    derivative_name: str,
    expr: sympy.Expr,
    taken: set[str],
) -> tuple[list[tuple[sympy.Symbol, sympy.Expr]], sympy.Expr]:
    """Factor shared subexpressions out of one state's linearized expression.

    This is the ``CSEStrategy.per_state`` strategy: one ``sympy.cse`` call per
    state, considering only that state's own diagonal Jacobian entry. It is
    no longer the default -- ``CSEStrategy.joint`` (``_joint_linearized_assignments``)
    is, since it costs the same or fewer operations on every measured backend
    -- but it stays available because it becomes the better choice again if
    the Python backend starts emitting ``del`` for a temporary once it is
    dead: a per-state temporary is provably dead at the end of that state's
    own update, where a joint one may still be needed by a later state.

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


def _joint_linearized_assignments(
    items: list[tuple[str, sympy.Expr]],
    taken: set[str],
) -> dict[str, tuple[list[tuple[sympy.Symbol, sympy.Expr]], sympy.Expr]]:
    """Factor shared subexpressions out of every state's linearized expression
    at once (``CSEStrategy.joint``).

    ``items`` is ``(state_name, diagonal_jacobian_entry)`` pairs, in the order
    those states will be emitted. One ``sympy.cse`` call runs across all of
    them together, so a subexpression shared *between* two states' entries is
    computed once rather than twice.

    A temporary shared between states x and y must literally appear in both
    of their expressions, so its free symbols lie in both states' dependency
    cones and are therefore already available by the time either ``dx_dt`` or
    ``dy_dt`` has been emitted. That makes the emission rule simple: emit each
    temporary immediately before the first (in ``items`` order) state whose
    linearized expression needs it -- directly, or transitively through
    another temporary. ``sympy.cse``'s replacement list is already in
    topological order (a temporary only ever references an earlier one), so
    "first state that needs it" can be resolved with one forward pass.

    Returns, per state name, the temporaries that must be emitted immediately
    before that state's ``d<state>_dt_linearized`` line (only those not
    already emitted for an earlier state in ``items``) and this state's
    reduced expression.
    """
    names = [name for name, _ in items]
    exprs = [expr for _, expr in items]

    def fresh():
        i = 0
        while True:
            name = f"_linearization_temp_{i}"
            if name not in taken:
                yield sympy.Symbol(name, real=True)
            i += 1

    replacements, reduced = sympy.cse(exprs, symbols=fresh(), optimizations="basic")

    temp_symbols = {symbol for symbol, _ in replacements}
    sub_expr_by_symbol = dict(replacements)
    # Each temporary's own direct dependencies among the other temporaries;
    # used below to take the transitive closure of "temporaries this state's
    # expression needs".
    direct_temp_deps = {
        symbol: sub_expr.free_symbols & temp_symbols for symbol, sub_expr in replacements
    }
    emission_order = {symbol: i for i, (symbol, _) in enumerate(replacements)}

    def transitively_needed(expr: sympy.Expr) -> set[sympy.Symbol]:
        needed = expr.free_symbols & temp_symbols
        frontier = set(needed)
        while frontier:
            frontier = set().union(*(direct_temp_deps[s] for s in frontier)) - needed
            needed |= frontier
        return needed

    already_emitted: set[sympy.Symbol] = set()
    result: dict[str, tuple[list[tuple[sympy.Symbol, sympy.Expr]], sympy.Expr]] = {}
    for name, expr in zip(names, reduced):
        to_emit = sorted(
            transitively_needed(expr) - already_emitted, key=lambda s: emission_order[s]
        )
        result[name] = ([(s, sub_expr_by_symbol[s]) for s in to_emit], expr)
        already_emitted |= set(to_emit)

    return result


def _linearization_plan(
    items: list[tuple[str, sympy.Expr]],
    taken: set[str],
    cse: CSEStrategy,
) -> dict[str, tuple[list[tuple[sympy.Symbol, sympy.Expr]], sympy.Expr]]:
    """Compute the temporaries to emit and the reduced expression to assign,
    per derivative name, for exactly the states about to be linearized (never
    for one that will fall back to forward Euler -- that would emit a
    temporary nothing uses).

    ``items`` keys by derivative name (e.g. ``"dx_dt"``, i.e. ``x.name`` for a
    ``StateDerivative`` ``x``) rather than state name, matching the naming
    ``_linearized_assignments`` has always used for ``per_state``.
    """
    if cse is CSEStrategy.joint:
        return _joint_linearized_assignments(items, taken)
    if cse is CSEStrategy.per_state:
        return {name: _linearized_assignments(name, expr, taken) for name, expr in items}
    # CSEStrategy.none: nothing factored out, the full expression stands as-is.
    return {name: ([], expr) for name, expr in items}


def _rush_larsen_update(
    x: atoms.StateDerivative,
    replacements: list[tuple[sympy.Symbol, sympy.Expr]],
    expr_diff: sympy.Expr,
    dt: sympy.Symbol,
    target: sympy.Symbol | sympy.Expr,
    printer: printer_func,
    delta: float,
) -> list[str]:
    """Emit the CSE temporaries, the linearized assignment, and the Rush-Larsen update.

    Shared by every scheme that reaches this point for a state whose diagonal
    Jacobian entry is non-zero. The zero-derivative case is each caller's own
    forward-Euler fallback, decided before this is called, since the two
    schemes reach it under different conditions.

    ``replacements`` and ``expr_diff`` (the post-CSE reduced expression) are
    computed by the caller, which decides -- once, for the whole scheme --
    which ``CSEStrategy`` produced them (``_linearized_assignments`` for
    ``per_state``, ``_joint_linearized_assignments`` for ``joint``, or
    ``([], expr_diff)`` unchanged for ``none``).
    """
    eqs = []
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
    eqs.append(printer(target, x.state.symbol + RL_term))
    return eqs


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
    cse: CSEStrategy | str = CSEStrategy.joint,
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
        Euler. For a state that can be linearized, this is a cost/accuracy
        trade-off, not a capability gate: the Rush-Larsen update costs an
        exponential per state per step. The exception is a state whose
        derivative does not depend on itself; it gets forward Euler regardless
        (the exact g -> 0 limit), and a warning is logged.
    cse : CSEStrategy | str, optional
        Strategy for factoring shared subexpressions out of the linearized
        expressions, by default ``CSEStrategy.joint``. See `CSEStrategy` for
        the trade-offs between ``"joint"``, ``"per_state"``, and ``"none"``.

    Returns
    -------
    list[str]
        A list of equations as strings

    """
    if stiff_states is None:
        stiff_states = []
    cse = CSEStrategy(cse)
    logger.debug("Generating hybrid Rush-Larsen scheme", stiff_states=stiff_states, cse=cse.value)
    stiff_states_set = set(stiff_states)
    found_stiff_states_set = set()
    not_linearizable = set()
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    taken = _taken_names(ode)

    # The states that will actually receive the Rush-Larsen update, in
    # emission order: stiff and linearizable. Anything else (not stiff, or
    # stiff but not linearizable) falls back to forward Euler and must not be
    # included -- CSE would factor a temporary nothing uses.
    to_linearize = [
        x
        for x in ode.sorted_assignments(remove_unused=remove_unused)
        if isinstance(x, atoms.StateDerivative)
        and x.state.name in stiff_states_set
        and not jacobian[x.state.name].is_zero
    ]
    plan = _linearization_plan([(x.name, jacobian[x.state.name]) for x in to_linearize], taken, cse)

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
        replacements, reduced = plan[x.name]
        eqs.extend(_rush_larsen_update(x, replacements, reduced, dt, values[i], printer, delta))
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
    cse: CSEStrategy | str = CSEStrategy.joint,
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
    cse : CSEStrategy | str, optional
        Strategy for factoring shared subexpressions out of the linearized
        expressions, by default ``CSEStrategy.joint``. See `CSEStrategy` for
        the trade-offs between ``"joint"``, ``"per_state"``, and ``"none"``.

    Returns
    -------
    list[str]
        A list of equations as strings
    """
    cse = CSEStrategy(cse)
    logger.debug("Generating generalized Rush-Larsen scheme", cse=cse.value)
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    taken = _taken_names(ode)

    # Every state that will actually be linearized (non-zero diagonal entry),
    # in emission order. A zero entry falls back to forward Euler below and
    # must not be included -- CSE would factor a temporary nothing uses.
    to_linearize = [
        x
        for x in ode.sorted_assignments(remove_unused=remove_unused)
        if isinstance(x, atoms.StateDerivative) and not jacobian[x.state.name].is_zero
    ]
    plan = _linearization_plan([(x.name, jacobian[x.state.name]) for x in to_linearize], taken, cse)

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

        replacements, reduced = plan[x.name]
        eqs.extend(_rush_larsen_update(x, replacements, reduced, dt, values[i], printer, delta))
        i += 1
    return eqs
