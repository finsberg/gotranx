from __future__ import annotations

import sympy
from structlog import get_logger

from . import atoms
from .ode import ODE

logger = get_logger()

__all__ = ["diagonal_jacobian"]


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
    jacobian: dict[str, sympy.Expr] = {}

    for state in ode.states:
        # Seed the tangent at this state: d(state)/d(state) = 1.
        tangent: dict[sympy.Symbol, sympy.Expr] = {state.symbol: sympy.Integer(1)}
        for assignment in assignments:
            derivative = sympy.Integer(0)
            for symbol in assignment.expr.free_symbols:
                seed = tangent.get(symbol)
                if seed is None or seed == 0:
                    # This symbol carries no dependence on `state`.
                    continue
                derivative += sympy.diff(assignment.expr, symbol) * seed
            tangent[assignment.symbol] = derivative

            if (
                isinstance(assignment, atoms.StateDerivative)
                and assignment.state.name == state.name
            ):
                jacobian[state.name] = derivative

        # `check_components` guarantees every state has a matching
        # `StateDerivative`, so the inner loop above always finds one. Enforce
        # the promise made in the docstring rather than relying on that
        # invariant holding forever: a caller indexing this dict by state name
        # (schemes.py) should get a defined zero, not a bare `KeyError`.
        jacobian.setdefault(state.name, sympy.Integer(0))

    logger.debug(
        "Computed diagonal Jacobian",
        num_states=len(jacobian),
        num_nonzero=sum(1 for v in jacobian.values() if not v.is_zero),
    )
    return jacobian
