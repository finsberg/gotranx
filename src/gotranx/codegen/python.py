from __future__ import annotations
import typing
from enum import Enum
from sympy.printing.pycode import PythonCodePrinter

# from sympy.printing.numpy import NumPyPrinter
from sympy.codegen.ast import Assignment
import sympy
import structlog
from functools import partial

from ..ode import ODE
from .. import templates
from .base import CodeGenerator, Func, RHSArgument, SchemeArgument, _print_Piecewise

logger = structlog.get_logger()


class Format(str, Enum):
    black = "black"
    ruff = "ruff"
    none = "none"


# class GotranPythonCodePrinter(NumPyPrinter):
class GotranPythonCodePrinter(PythonCodePrinter):
    _kf = {
        **{k: f"numpy.{v.replace('math.', '')}" for k, v in PythonCodePrinter._kf.items()},
        **{"DiracDelta": "numpy.zeros_like"},
    }
    _kc = {k: f"numpy.{v.replace('math.', '')}" for k, v in PythonCodePrinter._kc.items()}

    def _hprint_Pow(self, expr, rational=False, sqrt="numpy.sqrt"):
        return super()._hprint_Pow(expr, rational, sqrt)

    def _print_MatrixElement(self, expr):
        if expr.parent.shape[1] == 1:
            # Then this is a column vector
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

    # def _print_Equality(self, expr):
    #     lhs, rhs = expr.args
    #     return f"numpy.isclose({self._print(lhs)}, {self._print(rhs)})"

    # def _print_sign(self, e):
    #     return "(0.0 if numpy.isclose({e}, 0) else {f}(1, {e}))".format(
    #         f=self._module_format("numpy.copysign"), e=self._print(e.args[0])
    #     )

    def _print_Equality(self, expr):
        lhs, rhs = expr.args
        return f"({self._print(lhs)} == {self._print(rhs)})"

    def _print_sign(self, e):
        return "(0.0 if ({e} == 0) else {f}(1, {e}))".format(
            f=self._module_format("numpy.copysign"), e=self._print(e.args[0])
        )


def get_formatter(format: Format) -> typing.Callable[[str], str]:
    if format == Format.none:
        return lambda x: x

    elif format == Format.black:
        try:
            import black
        except ImportError:
            logger.debug("Cannot apply black, please install 'black'")
            return lambda x: x
        else:
            return partial(black.format_str, mode=black.Mode())

    elif format == Format.ruff:
        try:
            import ruff.__main__

            ruff_bin = ruff.__main__.find_ruff_bin()
        except ImportError:
            logger.debug("Cannot apply ruff, please install 'ruff'")
            return lambda x: x
        else:
            import subprocess

            def func(code: str) -> str:
                return subprocess.check_output(
                    [ruff_bin, "format", "-"], input=code, encoding="utf-8"
                )

            return func

    else:
        raise ValueError(f"Unknown format {format}")


class PythonCodeGenerator(CodeGenerator):
    def __init__(self, ode: ODE, format: Format = Format.black, *args, **kwargs) -> None:
        super().__init__(ode, *args, **kwargs)

        self._printer = GotranPythonCodePrinter()

        setattr(self, "_formatter", get_formatter(format=format))

    @property
    def printer(self):
        return self._printer

    @property
    def template(self):
        return templates.python

    def imports(self) -> str:
        return self._format("import numpy")

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
            values_type="numpy.zeros_like(states, dtype=numpy.float64)",
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
            values_type="numpy.zeros_like(states, dtype=numpy.float64)",
        )
