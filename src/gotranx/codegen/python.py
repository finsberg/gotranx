from __future__ import annotations
from sympy.printing.pycode import PythonCodePrinter
from sympy.codegen.ast import Assignment
import sympy
from functools import partial

from ..ode import ODE
from .. import templates
from .base import CodeGenerator, Func, RHSArgument, SchemeArgument


# class GotranPythonCodePrinter(NumPyPrinter):
class GotranPythonCodePrinter(PythonCodePrinter):
    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _print_MatrixElement(self, expr):
        if expr.parent.shape[1] == 1:
            # Then this is a colum vector
            return f"{self._print(expr.parent)}[{expr.i}]"
        elif expr.parent.shape[0] == 1:
            # Then this is a row vector
            return f"{self._print(expr.parent)}[{expr.j}]"
        else:
            return super()._print_MatrixElement(expr)

    def _print_Float(self, flt):
        return self._print(str(float(flt)))

    def _print_Piecewise(self, expr):
        fst, snd = expr.args
        if isinstance(fst[0], Assignment):
            value = (
                f"{super()._print(fst[0].args[0])} = "
                f"{super()._print(fst[0].args[1])} if "
                f"{super()._print(fst[1])} else "
                f"{super()._print(snd[0].args[1])}"
            )
        else:
            value = super()._print_Piecewise(expr)

        return value


def squeeze_list(lst):
    if len(lst) == 1:
        return lst[0]
    return lst


def squeeze_list_of_lists(lst):
    return [squeeze_list(item) for item in lst]


class PythonCodeGenerator(CodeGenerator):
    def __init__(self, ode: ODE, apply_black: bool = True) -> None:
        super().__init__(ode)

        self._printer = GotranPythonCodePrinter()

        if apply_black:
            try:
                import black
            except ImportError:
                print("Cannot apply black, please install 'black'")
            else:
                # TODO: add options for black in Mode
                setattr(self, "_formatter", partial(black.format_str, mode=black.Mode()))

    @property
    def printer(self):
        return self._printer

    @property
    def template(self):
        return templates.python

    def _rhs_arguments(
        self,
        order: RHSArgument | str = RHSArgument.stp,
    ) -> Func:
        value = RHSArgument.get_value(order)

        argument_dict = {
            "s": "states",
            "t": "t",
            "p": "parameters",
        }

        argument_list = [argument_dict[v] for v in value]
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,))
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,))
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,))

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            return_name="values",
            num_return_values=self.ode.num_states,
        )

    def _scheme_arguments(
        self,
        order: SchemeArgument | str = SchemeArgument.stdp,
    ) -> Func:
        value = SchemeArgument.get_value(order)

        argument_dict = {
            "s": "states",
            "t": "t",
            "d": "dt",
            "p": "parameters",
        }

        argument_list = [argument_dict[v] for v in value]
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,))
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,))
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,))

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            return_name="values",
            num_return_values=self.ode.num_states,
        )
