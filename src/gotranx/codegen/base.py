from __future__ import annotations

import abc
import typing
from enum import Enum

import sympy
from sympy.codegen.ast import Assignment
from sympy.printing.codeprinter import CodePrinter

from .. import templates
from ..ode import ODE
from ..sympy_ode import SympyODE


class RHS(typing.NamedTuple):
    arguments: list[str]
    states: sympy.MatrixSymbol
    parameters: sympy.MatrixSymbol
    values: sympy.MatrixSymbol
    return_name: str | None = None


class RHSArgument(str, Enum):
    stp = "stp"
    spt = "spt"
    tsp = "tsp"
    tps = "tps"
    pst = "pst"
    pts = "pts"

    @staticmethod
    def get_value(order: str | RHSArgument) -> str:
        if isinstance(order, RHSArgument):
            return order.value
        return str(order)


class CodeGenerator(abc.ABC):
    variable_prefix = ""

    def __init__(self, ode: ODE) -> None:
        self.ode = ode
        self.sympy_ode = SympyODE(ode)

    def _formatter(self, code: str) -> str:
        """Alternative formatter that takes a code snippet
        and output a formatted code snipped together with
        a potential list of errors

        Parameters
        ----------
        code : str
            The code snippet

        Returns
        -------
        str
            formatted code snippet
        """
        return code

    def _format(self, code: str) -> str:
        try:
            formatted_code = self._formatter(code)
        except Exception:
            # FIXME: handle this
            print("An exception was raised")
            formatted_code = code

        return formatted_code

    def state_index(self) -> str:
        code = self.template.state_index(data={s.name: i for i, s in enumerate(self.ode.states)})
        return self._format(code)

    def parameter_index(self) -> str:
        code = self.template.parameter_index(
            data={s.name: i for i, s in enumerate(self.ode.parameters)}
        )
        return self._format(code)

    def initial_state_values(self, name="states") -> str:
        state_result = sympy.MatrixSymbol(name, self.ode.num_states, 1)

        expr = self.printer.doprint(self.sympy_ode.state_values, assign_to=state_result)

        code = self.template.init_state_values(
            code=expr,
            state_names=[s.name for s in self.ode.states],
            state_values=[s.value for s in self.ode.states],
            name=name,
        )
        return self._format(code)

    def initial_parameter_values(self, name="parameters") -> str:
        parameter_result = sympy.MatrixSymbol(name, self.ode.num_parameters, 1)

        expr = self.printer.doprint(
            self.sympy_ode.parameter_values,
            assign_to=parameter_result,
        )
        code = self.template.init_parameter_values(
            code=expr,
            parameter_names=[s.name for s in self.ode.parameters],
            parameter_values=[s.value for s in self.ode.parameters],
            name=name,
        )
        return self._format(code)

    def _state_assignments(self, states: sympy.MatrixSymbol) -> str:
        return "\n".join(
            [
                f"{self.variable_prefix}{self.printer.doprint(Assignment(state.symbol, value))}"
                for state, value in zip(self.ode.states, states)
            ],
        )

    def _parameter_assignments(self, parameters: sympy.MatrixSymbol) -> str:
        return "\n".join(
            [
                f"{self.variable_prefix}{self.printer.doprint(Assignment(param.symbol, value))}"
                for param, value in zip(
                    self.ode.parameters,
                    parameters,
                )
            ],
        )

    def rhs(self, order: RHSArgument | str = RHSArgument.stp, use_cse=False) -> str:
        # breakpoint()
        rhs = self._rhs_arguments(order)

        states = self._state_assignments(rhs.states)
        parameters = self._parameter_assignments(rhs.parameters)

        # state_names = [x.name for x in self.ode.states]
        # values = []
        # for sym in self.ode.sorted_assignments:
        #     x = self.ode[sym]
        #     if isinstance(x, atoms.Intermediate):
        #         values.append(
        #             f"{self.variable_prefix}{self.printer.doprint(Assignment(x.symbol, x.expr))}",
        #         )

        #     elif isinstance(x, atoms.StateDerivative):
        #         values.append(
        #             self.printer.doprint(
        #                 Assignment(rhs.values[state_names.index(x.state.name)], x.expr),
        #             ),
        #         )
        #     else:
        #         raise RuntimeError("What?")

        if use_cse:
            replacements, reduced_exprs = sympy.cse(self.sympy_ode.rhs)
            values_lst = []
            for replacement in replacements:
                values_lst.append(
                    f"{self.variable_prefix}{self.printer.doprint(Assignment(*replacement))}",
                )
            values = "\n".join(values_lst) + self.printer.doprint(
                reduced_exprs[0], assign_to=rhs.values
            )

        else:
            values = self.printer.doprint(self.sympy_ode.rhs, assign_to=rhs.values)
        # expr = expr.xreplace({s: rhs.values[i] for i, s in enumerate(sym)})
        # values = self.printer.doprint(expr, assign_to=rhs.values)

        code = self.template.method(
            name="rhs",
            args=", ".join(rhs.arguments),
            states=states,
            parameters=parameters,
            values=values,
            return_name=rhs.return_name,
        )

        return self._format(code)

    # def forward_euler(self, order: RHSArgument | str = RHSArgument.stp):
    #     """
    #     .. math
    #         y_{n+1} = y_n + h f(t_n, y_n)
    #     """
    #     rhs = self._rhs_arguments(order)

    #     states = self._state_assignments(rhs.states)
    #     parameters = self._parameter_assignments(rhs.parameters)

    #     dt = sympy.Symbol("dt")
    #     values = self.sympy_ode.states + dt * self.sympy_ode.state_derivatives

    #     # states = states + h * dy_dt
    #     # args = "double *__restrict states, const double t,
    #     # const, double dt, const double *__restrict parameters"
    #     breakpoint()
    #     code = self.template.METHOD.format(
    #         name="forward_euler",
    #         args=", ".join(rhs.arguments + ["dt"]),
    #         states=states,
    #         parameters=parameters,
    #         values="\n".join(values),
    #     )
    #     print(code)
    #     # breakpoint()

    @property
    @abc.abstractmethod
    def printer(self) -> CodePrinter:
        ...

    @property
    @abc.abstractmethod
    def template(self) -> templates.Template:
        ...

    @abc.abstractmethod
    def _rhs_arguments(self, order: RHSArgument | str) -> RHS:
        ...
