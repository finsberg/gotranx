from __future__ import annotations
from sympy.printing.c import C99CodePrinter
from sympy.codegen.ast import Assignment
import sympy

from ..ode import ODE
from .. import templates
from .base import CodeGenerator, RHS, RHSArgument


class GotranCCodePrinter(C99CodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._settings["contract"] = False

    def _print_Float(self, flt):
        return self._print(str(float(flt)))

    def _print_Piecewise(self, expr):
        fst, snd = expr.args
        if isinstance(fst[0], Assignment):
            value = (
                f"{super()._print(fst[0].args[0])} = "
                f"({super()._print(fst[1])}) ? "
                f"{super()._print(fst[0].args[1])} : "
                f"{super()._print(snd[0].args[1])};"
            )
        else:
            value = super()._print_Piecewise(expr)

        return value


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

    def _rhs_arguments(
        self, order: RHSArgument | str = RHSArgument.stp, const_states: bool = True
    ) -> RHS:
        value = RHSArgument.get_value(order)
        states_prefix = "const " if const_states else ""
        argument_dict = {
            "s": states_prefix + "double *__restrict states",
            "t": "const double t",
            "p": "const double *__restrict parameters",
        }
        argument_list = [argument_dict[v] for v in value] + ["double* values"]
        states = sympy.MatrixSymbol("states", self.ode.num_states, 1)
        parameters = sympy.MatrixSymbol("parameters", self.ode.num_parameters, 1)
        values = sympy.MatrixSymbol("values", self.ode.num_states, 1)

        return RHS(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
        )
