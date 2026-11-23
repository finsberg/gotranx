from __future__ import annotations

import sympy
from structlog import get_logger

from . import atoms
from .ode import ODE

logger = get_logger()

__all__ = ["diagonal_jacobian", "gating_states"]


def _tangent_sweep(assignments, seed: sympy.Symbol) -> dict[sympy.Symbol, sympy.Expr]:
    """Forward-mode derivative of every assignment with respect to ``seed``.

    Returns the tangent of ``seed`` (one) and of every assignment symbol, in
    terms of the model's own symbols (intermediates are not expanded).
    """
    tangent: dict[sympy.Symbol, sympy.Expr] = {seed: sympy.Integer(1)}
    for assignment in assignments:
        derivative = sympy.Integer(0)
        for symbol in assignment.expr.free_symbols:
            s = tangent.get(symbol)
            if s is None or s == 0:
                # This symbol carries no dependence on `seed`.
                continue
            derivative += sympy.diff(assignment.expr, symbol) * s
        tangent[assignment.symbol] = derivative
    return tangent


def diagonal_jacobian(ode: ODE, remove_unused: bool = False) -> dict[str, sympy.Expr]:
    r"""Compute :math:`\partial f_i / \partial y_i` for every state.

    This is the diagonal of the Jacobian of the full right-hand side, which is
    the quantity the generalized Rush-Larsen method is defined in terms of. It
    differentiates *through* intermediate assignments, so it does not depend on
    whether the model author named a subexpression.

    Implemented as forward-mode automatic differentiation: one tangent sweep per
    state over the topologically sorted assignments, seeded at that state.

    Parameters
    ----------
    ode : gotranx.ode.ODE
        The ODE
    remove_unused : bool, optional
        Sweep only the assignments the state derivatives depend on, by default
        False. Should match the value passed to ``ode.sorted_assignments`` by
        the caller.

    Returns
    -------
    dict[str, sympy.Expr]
        Mapping from state *name* to its diagonal Jacobian entry. Every state of
        the ODE is present; an entry may be zero, which means the derivative
        genuinely does not depend on its own state.
    """
    assignments = ode.sorted_assignments(remove_unused=remove_unused)
    derivative_symbol = {
        a.state.name: a.symbol for a in assignments if isinstance(a, atoms.StateDerivative)
    }
    jacobian: dict[str, sympy.Expr] = {}
    for state in ode.states:
        tangent = _tangent_sweep(assignments, state.symbol)
        symbol = derivative_symbol.get(state.name)
        if symbol is not None:
            jacobian[state.name] = tangent[symbol]
        # `check_components` guarantees every state has a matching
        # `StateDerivative`. Enforce the docstring's promise anyway: a caller
        # indexing this dict by state name should get a defined zero, not a
        # bare `KeyError`.
        jacobian.setdefault(state.name, sympy.Integer(0))
    logger.debug(
        "Computed diagonal Jacobian",
        num_states=len(jacobian),
        num_nonzero=sum(1 for v in jacobian.values() if not v.is_zero),
    )
    return jacobian


def _dependency_cone(root: sympy.Symbol, by_symbol: dict) -> set[sympy.Symbol]:
    """``root`` and every assignment symbol it depends on, transitively."""
    cone = {root}
    stack = [root]
    while stack:
        assignment = by_symbol.get(stack.pop())
        if assignment is None:
            continue
        for symbol in assignment.expr.free_symbols:
            if symbol not in cone:
                cone.add(symbol)
                stack.append(symbol)
    return cone


def gating_states(ode: ODE, remove_unused: bool = False) -> list[str]:
    r"""The states whose right-hand side is affine in the state itself.

    These are the states the classic Rush-Larsen method integrates
    exponentially: :math:`f_i = a_i y_i + b_i` with :math:`a_i \ne 0` and
    :math:`a_i`, :math:`b_i` independent of :math:`y_i` (e.g. a
    Hodgkin-Huxley gate :math:`(m_\infty - m)/\tau_m`). A state qualifies when

    * its diagonal Jacobian entry :math:`a_i` is not zero,
    * :math:`\partial a_i / \partial y_i` is structurally zero, and
    * no conditional that :math:`f_i` depends on has a condition that depends
      on :math:`y_i` (differentiation ignores conditions, so a branch switch
      on :math:`y_i` would otherwise go unnoticed).

    Anything that cannot be shown to satisfy these is not gating.

    Parameters
    ----------
    ode : gotranx.ode.ODE
        The ODE
    remove_unused : bool, optional
        Passed on to ``ode.sorted_assignments``, by default False

    Returns
    -------
    list[str]
        The names of the gating states, in ``ode.states`` order
    """
    assignments = ode.sorted_assignments(remove_unused=remove_unused)
    by_symbol = {a.symbol: a for a in assignments}
    derivative_of = {a.state.name: a for a in assignments if isinstance(a, atoms.StateDerivative)}
    gating = []
    for state in ode.states:
        derivative = derivative_of.get(state.name)
        if derivative is None:
            continue
        tangent = _tangent_sweep(assignments, state.symbol)
        a = tangent[derivative.symbol]
        if a.is_zero:
            continue
        second = sympy.Integer(0)
        for symbol in a.free_symbols:
            s = tangent.get(symbol)
            if s is None or s == 0:
                continue
            second += sympy.diff(a, symbol) * s
        if second != 0:
            continue
        depends_on_state = {sym for sym, t in tangent.items() if t != 0}
        branches_on_state = False
        for symbol in _dependency_cone(derivative.symbol, by_symbol):
            assignment = by_symbol.get(symbol)
            if assignment is None:
                continue
            for piecewise in assignment.expr.atoms(sympy.Piecewise):
                for _, condition in piecewise.args:
                    if getattr(condition, "free_symbols", set()) & depends_on_state:
                        branches_on_state = True
        if branches_on_state:
            continue
        gating.append(state.name)
    logger.debug("Detected gating states", gating_states=gating)
    return gating
