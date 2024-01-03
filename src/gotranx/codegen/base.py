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


class Func(typing.NamedTuple):
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


class SchemeArgument(str, Enum):
    stpd = "stpd"
    sptd = "sptd"
    tspd = "tspd"
    tpsd = "tpsd"
    pstd = "pstd"
    ptsd = "ptsd"
    stdp = "stdp"
    spdt = "spdt"
    tsdp = "tsdp"
    tpds = "tpds"
    psdt = "psdt"
    ptds = "ptds"
    sdtp = "sdtp"
    sdpt = "sdpt"
    tdsp = "tdsp"
    tdps = "tdps"
    pdst = "pdst"
    pdts = "pdts"
    dstp = "dstp"
    dspt = "dspt"
    dtsp = "dtsp"
    dtps = "dtps"
    dpst = "dpst"
    dpts = "dpts"

    @staticmethod
    def get_value(order: str | SchemeArgument) -> str:
        if isinstance(order, SchemeArgument):
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
        code = self.template.state_index(
            data={s.name: i for i, s in enumerate(self.ode.sorted_states())}
        )
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
                for i, value in enumerate([s.value for s in self.ode.sorted_states()])
            ]
        )

        code = self.template.init_state_values(
            code=expr,
            state_names=[s.name for s in self.ode.sorted_states()],
            state_values=[self.printer.doprint(s.value) for s in self.ode.sorted_states()],
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
            parameter_values=[self.printer.doprint(s.value) for s in self.ode.parameters],
            name=name,
        )
        return self._format(code)

    def _state_assignments(self, states: sympy.IndexedBase) -> str:
        return "\n".join(
            self._doprint(state.symbol, states[i], use_variable_prefix=True)
            for i, state in enumerate(self.ode.sorted_states())
        )

    def _parameter_assignments(self, parameters: sympy.IndexedBase) -> str:
        return "\n".join(
            self._doprint(param.symbol, parameters[i], use_variable_prefix=True)
            for i, param in enumerate(self.ode.parameters)
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

        values_lst = []
        index = 0
        values_idx = sympy.IndexedBase("values", shape=(len(self.ode.state_derivatives),))

        for x in self.ode.sorted_assignments():
            values_lst.append(self._doprint(x.symbol, x.expr, use_variable_prefix=True))
            if isinstance(x, atoms.StateDerivative):
                values_lst.append(self._doprint(values_idx[index], x.symbol))
                index += 1

        values = "\n".join(values_lst)
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

    def scheme(self, name: str, order=SchemeArgument.stdp) -> str:
        """Generate code for the forward explicit Euler method

        Parameters
        ----------
        name : str
            The name of the scheme
        order : SchemeArgument | str, optional
            The order of the arguments, by default SchemeArgument.stdp

        Returns
        -------
        str
            The generated code
        """

        rhs = self._scheme_arguments(order)
        states = self._state_assignments(rhs.states)
        parameters = self._parameter_assignments(rhs.parameters)

        dt = sympy.Symbol("dt")
        f = schemes.get_scheme(name)
        eqs = f(self.ode, dt, name=rhs.return_name, printer=self._doprint)
        values = "\n".join(eqs)

        code = self.template.method(
            name=name,
            args=", ".join(rhs.arguments),
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
    def _rhs_arguments(self, order: RHSArgument | str) -> Func:
        ...

    @abc.abstractmethod
    def _scheme_arguments(self, order: SchemeArgument | str) -> Func:
        ...
