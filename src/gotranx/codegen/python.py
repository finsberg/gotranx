from __future__ import annotations
from sympy.printing.pycode import PythonCodePrinter

# from sympy.printing.numpy import NumPyPrinter
from sympy.codegen.ast import Assignment
import sympy
from functools import partial

from ..ode import ODE
from .. import templates
from .base import CodeGenerator, Func, RHSArgument, SchemeArgument, _print_Piecewise


# class GotranPythonCodePrinter(NumPyPrinter):
class GotranPythonCodePrinter(PythonCodePrinter):
    _kf = {k: f"numpy.{v.replace('math.', '')}" for k, v in PythonCodePrinter._kf.items()}
    _kc = {k: f"numpy.{v.replace('math.', '')}" for k, v in PythonCodePrinter._kc.items()}

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
        result = []

        if isinstance(expr.args[0][0], Assignment):
            lhs = super()._print(expr.args[0][0].lhs)
            result.append(f"{super()._print(lhs)} = ")
            all_lsh_equal = True
            for arg in expr.args:
                result.append("numpy.where(")
                result.append(f"{super()._print(arg[1])}")
                result.append(", ")
                result.append(f"{super()._print(arg[0].rhs)}")
                result.append(", ")
                all_lsh_equal = all_lsh_equal and super()._print(arg[0].lhs) == lhs

            assert all_lsh_equal, "All assignments in Piecewise must have the same lhs"

            if super()._print(arg[1]) == "True":
                result = result[:-6]
                result.append(f", {super()._print(arg[0].rhs)}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(")" * (len(expr.args) - 1))

        else:
            conds, exprs = _print_Piecewise(self, expr)

            for c, e in zip(conds, exprs):
                result.append("numpy.where(")
                result.append(f"{c}")
                result.append(", ")
                result.append(f"{e}")
                result.append(", ")

            if c == "True":
                result = result[:-6]
                result.append(f", {e}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(")" * (len(conds) - 1))

        return "".join(result)

    def _print_And(self, expr):
        if len(expr.args) == 2:
            value = f"numpy.logical_and({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        else:
            args = ", ".join(self._print(arg) for arg in expr.args)
            value = f"numpy.logical_and.reduce(({args}))"

        return value

    def _print_Or(self, expr):
        # value = super()._print_Or(expr)
        if len(expr.args) == 2:
            value = f"numpy.logical_or({self._print(expr.args[0])}, {self._print(expr.args[1])})"
        else:
            args = ", ".join(self._print(arg) for arg in expr.args)
            value = f"numpy.logical_or.reduce(({args}))"

        return value


class PythonCodeGenerator(CodeGenerator):
    def __init__(self, ode: ODE, apply_black: bool = True, remove_unused: bool = False) -> None:
        super().__init__(ode, remove_unused=remove_unused)

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
