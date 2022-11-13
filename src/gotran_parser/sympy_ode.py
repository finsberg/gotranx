from __future__ import annotations

import attr
import sympy
from sympy.codegen.ast import Assignment

from . import atoms
from .ode import ODE


@attr.s(frozen=True, slots=True)
class SympyODE:
    ode: ODE = attr.ib()
    sorted_states: list[atoms.State] = attr.ib(init=False, repr=False)
    sorted_parameters: list[atoms.State] = attr.ib(init=False, repr=False)
    sorted_state_derivatives: list[atoms.StateDerivative] = attr.ib(
        init=False,
        repr=False,
    )

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "sorted_states",
            sorted(self.ode.states, key=lambda x: x.name),
        )
        object.__setattr__(
            self,
            "sorted_parameters",
            sorted(self.ode.parameters, key=lambda x: x.name),
        )
        object.__setattr__(
            self,
            "sorted_state_derivatives",
            sorted(self.ode.state_derivatives, key=lambda x: x.name),
        )

    @property
    def states(self) -> sympy.Matrix:
        return sympy.Matrix([state.symbol for state in self.sorted_states])

    @property
    def state_values(self) -> sympy.Matrix:
        return sympy.Matrix(
            [sympy.Float(str(state.value)) for state in self.sorted_states],
        )

    @property
    def parameters(self) -> sympy.Matrix:
        return sympy.Matrix([parameter.symbol for parameter in self.sorted_parameters])

    @property
    def parameter_values(self) -> sympy.Matrix:
        return sympy.Matrix([parameter.value for parameter in self.sorted_parameters])

    @property
    def state_derivatives(self) -> sympy.Matrix:
        return sympy.Matrix([state.symbol for state in self.sorted_state_derivatives])

    @property
    def rhs(self) -> sympy.Matrix:
        intermediates = {x.symbol: x.expr for x in self.ode.intermediates}
        rhs = sympy.Matrix([state.expr for state in self.sorted_state_derivatives])
        return rhs.xreplace(intermediates)

    @property
    def jacobian(self):
        return self.rhs.jacobian(self.states)

    @property
    def expressions(self):
        expressions = []
        for assignment_name in self.ode.sorted_assignments:
            assignment = self.ode._lookup[assignment_name]
            expressions.append(Assignment(assignment.symbol, assignment.expr))
        return expressions
