from __future__ import annotations
import typing
from types import CodeType

import sympy
from structlog import get_logger

from . import atoms
from .ode import ODE
from .linearization import diagonal_jacobian, gating_states
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
    rush_larsen = "rush_larsen"
    generalized_rush_larsen_2 = "generalized_rush_larsen_2"
    rush_larsen_2 = "rush_larsen_2"
    hybrid_rush_larsen_2 = "hybrid_rush_larsen_2"


def get_scheme(scheme: str) -> scheme_func:
    """Get the scheme function from a string"""
    if scheme in ["forward_euler", "forward_explicit_euler", "euler", "explicit_euler"]:
        func = explicit_euler
    elif scheme in ["forward_generalized_rush_larsen", "generalized_rush_larsen"]:
        func = generalized_rush_larsen
    elif scheme in ["forward_rush_larsen", "rush_larsen"]:
        func = rush_larsen
    elif scheme == "hybrid_rush_larsen":
        func = hybrid_rush_larsen
    elif scheme == "generalized_rush_larsen_2":
        func = generalized_rush_larsen_2
    elif scheme == "rush_larsen_2":
        func = rush_larsen_2
    elif scheme == "hybrid_rush_larsen_2":
        func = hybrid_rush_larsen_2
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
    items: list[tuple[str, sympy.Expr]],
    taken: set[str],
    prefix: str = "_linearization_temp_",
) -> dict[str, tuple[list[tuple[sympy.Symbol, sympy.Expr]], sympy.Expr]]:
    """Factor shared subexpressions out of every state's linearized expression
    at once.

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
            name = f"{prefix}{i}"
            if name not in taken:
                yield sympy.Symbol(name, real=True)
            i += 1

    # Not `sympy.cse` directly: it hoists out of `Piecewise` branches, which
    # would compute a guarded singular expression unconditionally. See
    # `sympytools.cse_hiding_piecewise`.
    replacements, reduced = sympytools.cse_hiding_piecewise(
        exprs, symbols=fresh(), optimizations="basic"
    )

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
    cse: bool,
    prefix: str = "_linearization_temp_",
) -> dict[str, tuple[list[tuple[sympy.Symbol, sympy.Expr]], sympy.Expr]]:
    """Compute the temporaries to emit and the reduced expression to assign,
    per derivative name, for exactly the states about to be linearized (never
    for one that will fall back to forward Euler -- that would emit a
    temporary nothing uses).

    ``items`` keys by derivative name (e.g. ``"dx_dt"``, i.e. ``x.name`` for a
    ``StateDerivative`` ``x``) rather than state name.
    """
    if cse:
        return _linearized_assignments(items, taken, prefix)
    # Nothing factored out: the full expression stands as-is, so there are no
    # temporaries to emit.
    return {name: ([], expr) for name, expr in items}


class _NameAllocator:
    """Hand out identifiers that collide with nothing in the model, nor with
    anything handed out before (``V_mid`` taken -> ``V_mid_1``)."""

    def __init__(self, taken: typing.Iterable[str]) -> None:
        self.used: set[str] = set(taken)

    def reserve(self, names: typing.Iterable[str]) -> None:
        self.used |= set(names)

    def __call__(self, base: str) -> sympy.Symbol:
        name, k = base, 0
        while name in self.used:
            k += 1
            name = f"{base}_{k}"
        self.used.add(name)
        return sympy.Symbol(name, real=True)


def _exponential_update(
    *,
    base: sympy.Expr,
    rate: sympy.Expr,
    linearized: sympy.Symbol,
    expr_diff: sympy.Expr,
    replacements: list[tuple[sympy.Symbol, sympy.Expr]],
    dt: sympy.Expr,
    target: sympy.Symbol | sympy.Expr,
    printer: printer_func,
    delta: float,
    declare_target: bool = False,
) -> list[str]:
    """Emit the CSE temporaries, the linearized assignment and the update
    ``target = base + rate * (exp(linearized*dt) - 1)/linearized``.

    ``rate`` is ``f`` for a first-order step and the stage-1 half step, and
    the midpoint linearization ``f* + a*(y - y*)`` for the second-order full
    step. Where ``linearized`` could be zero the update is guarded and falls
    back to its exact ``linearized -> 0`` limit ``base + dt*rate``.

    ``expr_diff`` is the post-CSE reduced expression, so the non-zero check
    runs on the CSE'd form: a provably-nonzero factor hidden behind an opaque
    temporary is conservatively guarded. That is an accepted cost of
    factoring first, not a bug.
    """
    eqs = [printer(symbol, sub_expr, use_variable_prefix=True) for symbol, sub_expr in replacements]
    eqs.append(printer(linearized, expr_diff, use_variable_prefix=True))

    need_zero_div_check = not fraction_numerator_is_nonzero(expr_diff)
    if not need_zero_div_check:
        logger.debug(f"{linearized.name} cannot be zero. Skipping zero division check")

    RL_term = rate / linearized * (sympy.exp(linearized * dt) - 1)
    if need_zero_div_check:
        RL_term = sympytools.Conditional(abs(linearized) > delta, RL_term, dt * rate)
    eqs.append(printer(target, base + RL_term, use_variable_prefix=declare_target))
    return eqs


def _rush_larsen_scheme(
    ode: ODE,
    dt: sympy.Symbol,
    *,
    linearize: set[str],
    order: int,
    name: str,
    printer: printer_func,
    remove_unused: bool,
    delta: float,
    cse: bool,
    jacobian: dict[str, sympy.Expr],
) -> list[str]:
    """Every Rush-Larsen scheme: states in ``linearize`` (with a non-zero
    diagonal Jacobian entry) get the exponential update, all others the
    ``a -> 0`` limit (forward Euler at order 1, explicit midpoint at order 2).
    """
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    assignments = ode.sorted_assignments(remove_unused=remove_unused)
    taken = _taken_names(ode)
    derivatives = [x for x in assignments if isinstance(x, atoms.StateDerivative)]
    to_linearize = [
        x for x in derivatives if x.state.name in linearize and not jacobian[x.state.name].is_zero
    ]
    linearized_names = {x.name for x in to_linearize}
    plan = _linearization_plan([(x.name, jacobian[x.state.name]) for x in to_linearize], taken, cse)

    if order == 1:
        stage1_dt = dt
        target: dict[str, sympy.Symbol | sympy.Expr] = {
            x.name: values[i] for i, x in enumerate(derivatives)
        }
        declare = False
    else:
        names = _NameAllocator(taken)
        names.reserve(x.name + "_linearized" for x in to_linearize)
        names.reserve(s.name for replacements, _ in plan.values() for s, _ in replacements)
        midpoint = {x.state.symbol: names(x.state.name + "_mid") for x in derivatives}
        stage1_dt = dt / 2
        target = {x.name: midpoint[x.state.symbol] for x in derivatives}
        declare = True

    eqs = []
    for x in assignments:
        eqs.append(printer(x.symbol, x.expr, use_variable_prefix=True))
        if not isinstance(x, atoms.StateDerivative):
            continue
        if x.name in linearized_names:
            replacements, reduced = plan[x.name]
            eqs.extend(
                _exponential_update(
                    base=x.state.symbol,
                    rate=x.symbol,
                    linearized=sympy.Symbol(x.name + "_linearized"),
                    expr_diff=reduced,
                    replacements=replacements,
                    dt=stage1_dt,
                    target=target[x.name],
                    printer=printer,
                    delta=delta,
                    declare_target=declare,
                )
            )
        else:
            logger.debug(f"{x.state.name} is not linearized; using the a -> 0 limit")
            eqs.append(
                printer(
                    target[x.name],
                    x.state.symbol + stage1_dt * x.symbol,
                    use_variable_prefix=declare,
                )
            )
    if order == 1:
        return eqs

    # Stage 2: the whole model again at (t + dt/2, y*), every assignment
    # under a fresh name, then the full step from y with the linearization
    # taken at the midpoint. Parameters and missing variables are unchanged.
    substitution: dict[sympy.Basic, sympy.Expr] = dict(midpoint)
    substitution[ode.t] = ode.t + dt / 2
    for x in assignments:
        substitution[x.symbol] = names(x.name + "_mid")
    linearized2 = {x.name: names(x.name + "_mid_linearized") for x in to_linearize}
    plan2 = _linearization_plan(
        [
            (substitution[x.symbol].name, jacobian[x.state.name].xreplace(substitution))
            for x in to_linearize
        ],
        names.used,
        cse,
        prefix="_linearization_mid_temp_",
    )

    i = 0
    for x in assignments:
        stage2_symbol = substitution[x.symbol]
        eqs.append(printer(stage2_symbol, x.expr.xreplace(substitution), use_variable_prefix=True))
        if not isinstance(x, atoms.StateDerivative):
            continue
        y = x.state.symbol
        if x.name in linearized_names:
            replacements, reduced = plan2[stage2_symbol.name]
            a_mid = linearized2[x.name]
            eqs.extend(
                _exponential_update(
                    base=y,
                    rate=stage2_symbol + a_mid * (y - midpoint[y]),
                    linearized=a_mid,
                    expr_diff=reduced,
                    replacements=replacements,
                    dt=dt,
                    target=values[i],
                    printer=printer,
                    delta=delta,
                )
            )
        else:
            eqs.append(printer(values[i], y + dt * stage2_symbol))
        i += 1
    return eqs


def _hybrid_linearize(
    ode: ODE,
    jacobian: dict[str, sympy.Expr],
    stiff_states: list[str] | None,
    remove_unused: bool,
    fallback: str,
) -> set[str]:
    """The stiff states to linearize, with the hybrid schemes' warnings."""
    stiff = set(stiff_states or [])
    found = {
        x.state.name
        for x in ode.sorted_assignments(remove_unused=remove_unused)
        if isinstance(x, atoms.StateDerivative) and x.state.name in stiff
    }
    not_linearizable = {s for s in found if jacobian[s].is_zero}
    if not_linearizable:
        logger.warning(
            "The following states were marked as stiff but their derivative does "
            f"not depend on the state, so they use {fallback}: "
            f"{sorted(not_linearizable)}"
        )
    missing = stiff.difference(found)
    if missing:
        logger.warning(
            f"The following states were marked as stiff but not found in the ODE: {sorted(missing)}"
        )
    return found - not_linearizable


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
    cse: bool = True,
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
    cse : bool, optional
        Factor shared subexpressions out of the linearized expressions with
        one ``sympy.cse`` pass across every linearized state at once, by
        default True. This costs far fewer operations than inlining them
        (1.08x the rhs against 9.43x, on ToRORd_dyn_chloride) at the price of
        one named local per factored subexpression. Pass False to inline
        instead, which emits no temporaries at all -- worth it only on the
        vectorized numpy backend, where each of those locals is a live array
        held until the generated function returns.

    Returns
    -------
    list[str]
        A list of equations as strings

    """
    logger.debug("Generating hybrid Rush-Larsen scheme", stiff_states=stiff_states, cse=cse)
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    linearize = _hybrid_linearize(ode, jacobian, stiff_states, remove_unused, "forward Euler")
    return _rush_larsen_scheme(
        ode,
        dt,
        linearize=linearize,
        order=1,
        name=name,
        printer=printer,
        remove_unused=remove_unused,
        delta=delta,
        cse=cse,
        jacobian=jacobian,
    )


def _gating_linearize(ode: ODE, remove_unused: bool, scheme: str, fallback: str) -> set[str]:
    gating = gating_states(ode, remove_unused=remove_unused)
    if not gating:
        logger.warning(
            f"No state of the ODE is affine in itself, so {scheme} integrates "
            f"every state with {fallback}"
        )
    return set(gating)


def rush_larsen(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
    cse: bool = True,
) -> list[str]:
    r"""Generate the (first-order) Rush-Larsen scheme for the ODE

    The exponential update of :func:`generalized_rush_larsen` is applied to
    the gating states, i.e. the states whose right-hand side is affine in the
    state itself (see :func:`gotranx.linearization.gating_states`); every
    other state uses forward Euler. Use :func:`hybrid_rush_larsen` to choose
    the states yourself.

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
    cse : bool, optional
        Factor shared subexpressions out of the linearized expressions, by
        default True (see :func:`generalized_rush_larsen`)

    Returns
    -------
    list[str]
        A list of equations as strings
    """
    logger.debug("Generating Rush-Larsen scheme", cse=cse)
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    linearize = _gating_linearize(ode, remove_unused, "rush_larsen", "forward Euler")
    return _rush_larsen_scheme(
        ode,
        dt,
        linearize=linearize,
        order=1,
        name=name,
        printer=printer,
        remove_unused=remove_unused,
        delta=delta,
        cse=cse,
        jacobian=jacobian,
    )


def generalized_rush_larsen_2(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
    cse: bool = True,
) -> list[str]:
    r"""Generate the second-order generalized Rush-Larsen scheme for the ODE

    With :math:`a_i = \partial f_i / \partial y_i` and
    :math:`\varphi(a, h) = (e^{a h} - 1)/a`, one step is

    .. math::
        y^*_i &= y_i + f_i(t_n, y)\, \varphi(a_i(t_n, y), \Delta t/2) \\
        y^{n+1}_i &= y_i + \left[f_i^* + a_i^* (y_i - y^*_i)\right] \varphi(a_i^*, \Delta t)

    where :math:`f^*` and :math:`a^*` are evaluated at :math:`(t_n + \Delta t/2, y^*)`.
    The bracket is the linearization of :math:`f_i` around :math:`y^*`
    evaluated at :math:`y_n`, which makes the step exact for a gate with
    frozen coefficients (Sundnes et al. 2009, Perego & Veneziani 2009).
    States with a zero diagonal entry get explicit midpoint.

    This differs from goss's ``GRL2``, which uses :math:`f(y^*)` as the
    numerator and evaluates the second stage at :math:`t_n`; that variant
    converges at first order on stiff, non-autonomous problems.

    Missing variables (values supplied by the caller) are held constant over
    the step. The generated code evaluates the model twice, so it is about
    twice the size of :func:`generalized_rush_larsen`.

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
    cse : bool, optional
        Factor shared subexpressions out of the linearized expressions of each
        stage, by default True (see :func:`generalized_rush_larsen`)

    Returns
    -------
    list[str]
        A list of equations as strings
    """
    logger.debug("Generating second-order generalized Rush-Larsen scheme", cse=cse)
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    return _rush_larsen_scheme(
        ode,
        dt,
        linearize=set(jacobian),
        order=2,
        name=name,
        printer=printer,
        remove_unused=remove_unused,
        delta=delta,
        cse=cse,
        jacobian=jacobian,
    )


def rush_larsen_2(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
    cse: bool = True,
) -> list[str]:
    r"""Generate the second-order Rush-Larsen scheme for the ODE

    The two-stage update of :func:`generalized_rush_larsen_2` applied to the
    gating states (see :func:`gotranx.linearization.gating_states`); every
    other state uses explicit midpoint. Use :func:`hybrid_rush_larsen_2` to
    choose the states yourself.

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
    cse : bool, optional
        Factor shared subexpressions out of the linearized expressions of each
        stage, by default True

    Returns
    -------
    list[str]
        A list of equations as strings
    """
    logger.debug("Generating second-order Rush-Larsen scheme", cse=cse)
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    linearize = _gating_linearize(ode, remove_unused, "rush_larsen_2", "explicit midpoint")
    return _rush_larsen_scheme(
        ode,
        dt,
        linearize=linearize,
        order=2,
        name=name,
        printer=printer,
        remove_unused=remove_unused,
        delta=delta,
        cse=cse,
        jacobian=jacobian,
    )


def hybrid_rush_larsen_2(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
    cse: bool = True,
) -> list[str]:
    r"""Generate the second-order hybrid Rush-Larsen scheme for the ODE

    The two-stage update of :func:`generalized_rush_larsen_2` applied to
    ``stiff_states``; every other state uses explicit midpoint, so with no
    stiff states this is the explicit midpoint (RK2) method. A stiff state
    whose derivative does not depend on itself also gets explicit midpoint,
    and a warning is logged.

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
    stiff_states : list[str] | None, optional
        States to integrate with the exponential update, by default None
    cse : bool, optional
        Factor shared subexpressions out of the linearized expressions of each
        stage, by default True

    Returns
    -------
    list[str]
        A list of equations as strings
    """
    logger.debug(
        "Generating second-order hybrid Rush-Larsen scheme", stiff_states=stiff_states, cse=cse
    )
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    linearize = _hybrid_linearize(ode, jacobian, stiff_states, remove_unused, "explicit midpoint")
    return _rush_larsen_scheme(
        ode,
        dt,
        linearize=linearize,
        order=2,
        name=name,
        printer=printer,
        remove_unused=remove_unused,
        delta=delta,
        cse=cse,
        jacobian=jacobian,
    )


def generalized_rush_larsen(
    ode: ODE,
    dt: sympy.Symbol,
    name: str = "values",
    printer: printer_func = default_printer,
    remove_unused: bool = False,
    delta: float = 1e-8,
    cse: bool = True,
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
    cse : bool, optional
        Factor shared subexpressions out of the linearized expressions with
        one ``sympy.cse`` pass across every linearized state at once, by
        default True. This costs far fewer operations than inlining them
        (1.08x the rhs against 9.43x, on ToRORd_dyn_chloride) at the price of
        one named local per factored subexpression. Pass False to inline
        instead, which emits no temporaries at all -- worth it only on the
        vectorized numpy backend, where each of those locals is a live array
        held until the generated function returns.

    Returns
    -------
    list[str]
        A list of equations as strings
    """
    logger.debug("Generating generalized Rush-Larsen scheme", cse=cse)
    jacobian = diagonal_jacobian(ode, remove_unused=remove_unused)
    return _rush_larsen_scheme(
        ode,
        dt,
        linearize=set(jacobian),
        order=1,
        name=name,
        printer=printer,
        remove_unused=remove_unused,
        delta=delta,
        cse=cse,
        jacobian=jacobian,
    )
