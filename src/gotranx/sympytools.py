from __future__ import annotations

import sympy

from . import atoms
from .ode import ODE


def forward_explicit_euler(ode: ODE, dt: sympy.Symbol, name: str = "values") -> list[sympy.Eq]:
    """Generate forward Euler equations for the ODE"""
    eqs = []
    values = sympy.IndexedBase(name, shape=(len(ode.state_derivatives),))
    i = 0
    for x in ode.sorted_assignments():
        # x: atoms.StateDerivative | atoms.Intermediate = ode[sym]
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


# def forward_generalized_rush_larsen(ode: ODE, dt: sympy.Symbol) -> list[sympy.Eq]:
#     """Generate forward Generalized Rush Larsen equations for the ODE"""
#     eqs = []
#     states = sympy.IndexedBase("states", shape=(len(ode.state_derivatives),))
#     for sym in ode.sorted_assignments:
#         x = ode[sym]
#         if isinstance(x, atoms.StateDerivative):
#             expr_diff = x.expr.diff(x.state.symbol)
#             breakpoint()
#             if expr_diff.is_zero:
#                 eqs.append(
#                     sympy.Eq(
#                         states[i],
#                         x.state.symbol + dt * x.symbol,
#                     )
#                 )
#         else:
#             eqs.append(sympy.Eq(x.symbol, x.expr))
#     return eqs


def states_matrix(ode: ODE) -> sympy.Matrix:
    return sympy.Matrix([state_der.state.symbol for state_der in ode.state_derivatives])


def rhs_matrix(ode: ODE, max_tries: int = 20) -> sympy.Matrix:
    intermediates = {x.symbol: x.expr for x in ode.intermediates}
    rhs = sympy.Matrix([state.expr for state in ode.state_derivatives])

    num_tries = 0
    while (any([rhs.has(k) for k in intermediates.keys()])) and num_tries < max_tries:
        rhs = rhs.xreplace(intermediates)
        num_tries += 1

    if num_tries == max_tries:
        raise RuntimeError("Maximum number of tries used")
    return rhs


def jacobi_matrix(ode: ODE) -> sympy.Matrix:
    return rhs_matrix(ode).jacobian(states_matrix(ode))
