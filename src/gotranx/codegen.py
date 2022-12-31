from __future__ import annotations

import abc
import typing
from enum import Enum

import sympy
from sympy.codegen.ast import Assignment
from sympy.printing.c import C99CodePrinter
from sympy.printing.codeprinter import CodePrinter

from . import templates
from .ode import ODE
from .sympy_ode import SympyODE


class RHS(typing.NamedTuple):
    arguments: str
    states: sympy.MatrixSymbol
    parameters: sympy.MatrixSymbol
    values: sympy.MatrixSymbol


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


class GotranCCodePrinter(C99CodePrinter):
    def _print_Float(self, flt):
        return self._print(str(float(flt)))


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

    def initial_state_values(self, name="states") -> str:
        state_result = sympy.MatrixSymbol(name, self.ode.num_states, 1)

        values = ", ".join(
            [f"{state.name}={state.value}" for state in self.sympy_ode.sorted_states],
        )
        expr = self.printer.doprint(self.sympy_ode.state_values, assign_to=state_result)

        code = self.template.INIT_STATE_VALUES.format(
            code=expr,
            values=values,
            name=name,
        )
        return self._format(code)

    def initial_parameter_values(self, name="parameters") -> str:
        parameter_result = sympy.MatrixSymbol(name, self.ode.num_parameters, 1)
        values = ", ".join(
            [
                f"{parameter.name}={parameter.value}"
                for parameter in self.sympy_ode.sorted_parameters
            ],
        )
        expr = self.printer.doprint(
            self.sympy_ode.parameter_values,
            assign_to=parameter_result,
        )
        code = self.template.INIT_PARAMETER_VALUES.format(
            code=expr,
            values=values,
            name=name,
        )
        return self._format(code)

    def rhs(self, order: RHSArgument | str = RHSArgument.stp) -> str:
        rhs = self._rhs_arguments(order)

        states = "\n".join(
            [
                f"{self.variable_prefix}{self.printer.doprint(Assignment(state.symbol, value))}"
                for state, value in zip(self.sympy_ode.sorted_states, rhs.states)
            ],
        )
        parameters = "\n".join(
            [
                f"{self.variable_prefix}{self.printer.doprint(Assignment(param.symbol, value))}"
                for param, value in zip(
                    self.sympy_ode.sorted_parameters,
                    rhs.parameters,
                )
            ],
        )
        values = self.printer.doprint(self.sympy_ode.rhs, assign_to=rhs.values)

        code = self.template.RHS.format(
            args=rhs.arguments,
            states=states,
            parameters=parameters,
            values=values,
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


class CCodeGenerator(CodeGenerator):
    variable_prefix = "double "

    def __init__(self, ode: ODE, apply_clang_format: bool = True) -> None:
        super().__init__(ode)
        self._printer = GotranCCodePrinter()

        if apply_clang_format:
            try:
                import clang_format_docs
            except ImportError:
                print("Cannot apply clang-format, please install 'clang-format-docs'")
            else:
                setattr(self, "_formatter", clang_format_docs.clang_format_str)

    @property
    def printer(self):
        return self._printer

    @property
    def template(self):
        return templates.c

    def _rhs_arguments(self, order: RHSArgument | str = RHSArgument.stp) -> RHS:
        value = RHSArgument.get_value(order)
        argument_dict = {
            "s": "const double *__restrict states",
            "t": "const double t",
            "p": "const double *__restrict parameters",
        }
        argument_list = [argument_dict[v] for v in value] + ["double* values"]
        states = sympy.MatrixSymbol("states", self.ode.num_states, 1)
        parameters = sympy.MatrixSymbol("parameters", self.ode.num_parameters, 1)
        values = sympy.MatrixSymbol("values", self.ode.num_states, 1)
        return RHS(
            arguments=", ".join(argument_list),
            states=states,
            parameters=parameters,
            values=values,
        )
