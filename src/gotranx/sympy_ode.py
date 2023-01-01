from __future__ import annotations

import attr
import sympy

from . import atoms
from .ode import ODE

# from sympy.codegen.ast import Assignment


@attr.s(frozen=True, slots=True)
class SympyODE:
    ode: ODE = attr.ib()

    @property
    def states(self) -> sympy.Matrix:
        return sympy.Matrix(
            [state_der.state.symbol for state_der in self.ode.state_derivatives],
        )

    @property
    def state_values(self) -> sympy.Matrix:
        return sympy.Matrix(
            [
                sympy.Float(str(state_der.state.value))
                for state_der in self.ode.state_derivatives
            ],
        )

    @property
    def parameters(self) -> sympy.Matrix:
        return sympy.Matrix([parameter.symbol for parameter in self.ode.parameters])

    @property
    def parameter_values(self) -> sympy.Matrix:
        return sympy.Matrix(
            [sympy.Float(str(parameter.value)) for parameter in self.ode.parameters],
        )

    @property
    def state_derivatives(self) -> sympy.Matrix:
        return sympy.Matrix([state.symbol for state in self.ode.state_derivatives])

    @property
    def rhs(self) -> sympy.Matrix:

        intermediates = {x.symbol: x.expr for x in self.ode.intermediates}
        rhs = sympy.Matrix([state.expr for state in self.ode.state_derivatives])

        return rhs.xreplace(intermediates)

    def rhs_sorted(self):
        syms_st = []
        expr_st = []
        syms_i = []
        expr_i = []

        for sym in self.ode.sorted_assignments:
            print(sym)

            x = self.ode[sym]
            if isinstance(x, atoms.Intermediate):
                syms_i.append(x.symbol)
                expr_i.append(x.expr)
            elif isinstance(x, atoms.StateDerivative):
                syms_st.append(x.symbol)
                expr_st.append(x.expr)
            else:
                raise RuntimeError("What?")

        # rhs = sympy.Matrix(expr_st)
        # breakpoint()
        # return syms_st, rhs.xreplace(dict(zip(syms_i, expr_i)))
        return syms_st, expr_st, syms_i, expr_i

    @property
    def jacobian(self):
        return self.rhs.jacobian(self.states)

    # @property
    # def expressions(self):
    #     expressions = []
    #     for assignment_name in self.ode.sorted_assignments:
    #         assignment = self.ode._lookup[assignment_name]
    #         expressions.append(Assignment(assignment.symbol, assignment.expr))
    #     return expressions
