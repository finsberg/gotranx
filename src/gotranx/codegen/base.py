from __future__ import annotations

import abc
import typing
from enum import Enum

import sympy
from sympy.codegen.ast import Assignment
from sympy.printing.codeprinter import CodePrinter

from .. import templates
from ..ode import ODE
from .. import atoms
from .. import schemes


class RHS(typing.NamedTuple):
    arguments: list[str]
    states: sympy.IndexedBase
    parameters: sympy.IndexedBase
    values: sympy.IndexedBase
    return_name: str = "values"
    num_return_values: int = 0


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

    def _doprint(self, lhs, rhs, use_variable_prefix: bool = False) -> str:
        if use_variable_prefix:
            return f"{self.variable_prefix}{self.printer.doprint(Assignment(lhs, rhs))}"
        return self.printer.doprint(Assignment(lhs, rhs))

    def state_index(self) -> str:
        code = self.template.state_index(data={s.name: i for i, s in enumerate(self.ode.states)})
        return self._format(code)

    def parameter_index(self) -> str:
        code = self.template.parameter_index(
            data={s.name: i for i, s in enumerate(self.ode.parameters)}
        )
        return self._format(code)

    def initial_state_values(self, name="states") -> str:
        """Generate code for initializing state values

        Parameters
        ----------
        name : str, optional
            The name of the variable, by default "states"

        Returns
        -------
        str
            The generated code
        """
        state_result = sympy.IndexedBase(name, shape=(self.ode.num_states,))
        expr = "\n".join(
            [
                self._doprint(state_result[i], value)
                for i, value in enumerate([s.state.value for s in self.ode.state_derivatives])
            ]
        )

        code = self.template.init_state_values(
            code=expr,
            state_names=[s.name for s in self.ode.states],
            state_values=[s.value for s in self.ode.states],
            name=name,
        )
        return self._format(code)

    def initial_parameter_values(self, name="parameters") -> str:
        """Generate code for initializing parameter values

        Parameters
        ----------
        name : str, optional
            The name of the variable, by default "parameters"

        Returns
        -------
        str
            The generated code
        """
        parameter_result = sympy.IndexedBase(name, shape=(self.ode.num_parameters,))

        expr = "\n".join(
            [
                self._doprint(parameter_result[i], value)
                for i, value in enumerate([p.value for p in self.ode.parameters])
            ]
        )
        code = self.template.init_parameter_values(
            code=expr,
            parameter_names=[s.name for s in self.ode.parameters],
            parameter_values=[s.value for s in self.ode.parameters],
            name=name,
        )
        return self._format(code)

    def _state_assignments(self, states: sympy.IndexedBase) -> str:
        return "\n".join(
            self._doprint(state.symbol, value, use_variable_prefix=True)
            for state, value in zip(self.ode.states, states)
        )

    def _parameter_assignments(self, parameters: sympy.IndexedBase) -> str:
        return "\n".join(
            self._doprint(param.symbol, value, use_variable_prefix=True)
            for param, value in zip(self.ode.parameters, parameters)
        )

    def rhs(self, order: RHSArgument | str = RHSArgument.tsp, use_cse=False) -> str:
        """Generate code for the right hand side of the ODE

        Parameters
        ----------
        order : RHSArgument | str, optional
            The order of the arguments, by default RHSArgument.tsp
        use_cse : bool, optional
            Use common subexpression elimination, by default False

        Returns
        -------
        str
            The generated code
        """

        rhs = self._rhs_arguments(order)
        states = self._state_assignments(rhs.states)
        parameters = self._parameter_assignments(rhs.parameters)

        # lhs_lst = []
        # rhs_lst = []
        values_lst = []
        index = 0
        values_idx = sympy.IndexedBase("values", shape=(len(self.ode.state_derivatives),))

        for x in self.ode.sorted_assignments():
            # rhs_lst.append(x.expr)
            if isinstance(x, atoms.Intermediate):
                # lhs_lst.append(x.symbol)
                values_lst.append(
                    f"{self.variable_prefix}"
                    f"{self.printer.doprint(Assignment(x.symbol, x.expr))}"
                )
            elif isinstance(x, atoms.StateDerivative):
                # lhs_lst.append(values_idx[index])
                values_lst.append(f"{self.printer.doprint(Assignment(values_idx[index], x.expr))}")
                index += 1
            else:
                raise RuntimeError(f"Unknown type {x}")

        values = "\n".join(values_lst)
        # breakpoint()
        code = self.template.method(
            name="rhs",
            args=", ".join(rhs.arguments),
            states=states,
            parameters=parameters,
            values=values,
            return_name=rhs.return_name,
            num_return_values=rhs.num_return_values,
        )

        return self._format(code)

    def scheme(self, name: str) -> str:
        """Generate code for the forward explicit Euler method

        Returns
        -------
        str
            The generated code
        """

        rhs = self._rhs_arguments(RHSArgument.stp)
        states = self._state_assignments(rhs.states)
        parameters = self._parameter_assignments(rhs.parameters)

        dt = sympy.Symbol("dt")
        f = schemes.get_scheme(name)
        eqs = f(self.ode, dt, name=rhs.return_name, printer=self._doprint)
        values = "\n".join(eqs)

        code = self.template.method(
            name=name,
            args=", ".join(rhs.arguments + ["dt"]),
            states=states,
            parameters=parameters,
            values=values,
            return_name=rhs.return_name,
            num_return_values=rhs.num_return_values,
        )
        return self._format(code)

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
