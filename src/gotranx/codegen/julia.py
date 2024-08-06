from __future__ import annotations
import structlog
from sympy.printing.julia import JuliaCodePrinter
from sympy.codegen.ast import Assignment
import sympy

from ..ode import ODE
from .. import templates
from .base import CodeGenerator, Func, RHSArgument, SchemeArgument

logger = structlog.get_logger()


# class Format(str, enum.Enum):
#     clang_format = "clang-format"
#     none = "none"


# def get_formatter(format: Format) -> typing.Callable[[str], str]:
#     if format == Format.none:
#         return lambda x: x
#     elif format == Format.clang_format:
#         try:
#             import clang_format_docs
#         except ImportError:
#             logger.warning("Cannot apply clang-format, please install 'clang-format-docs'")
#             return lambda x: x
#         else:
#             return clang_format_docs.clang_format_str
#     else:
#         raise ValueError(f"Unknown format: {format}")


def bool_to_int(expr: str) -> str:
    return expr.replace("false", "0").replace("true", "1")


class GotranJuliaCodePrinter(JuliaCodePrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._settings["contract"] = False

    def _print_Float(self, flt):
        return self._print(str(float(flt)))

    def _print_Piecewise(self, expr):
        if isinstance(expr.args[0][0], Assignment):
            result = []
            lhs = super()._print(expr.args[0][0].lhs)
            result.append(f"{super()._print(lhs)} = ")
            all_lsh_equal = True
            for arg in expr.args:
                result.append(f"({super()._print(arg[1])}) ? ")
                result.append(f"{super()._print(arg[0].rhs)}")
                result.append(" : ")
                all_lsh_equal = all_lsh_equal and super()._print(arg[0].lhs) == lhs

            assert all_lsh_equal, "All assignments in Piecewise must have the same lhs"

            if super()._print(arg[1]) == "true":
                result = result[:-3]
                result.append(f"{super()._print(arg[0].rhs)}")
            else:
                raise ValueError("Last condition in Piecewise must be True")

            result.append(";")
            value = "".join(result)
        else:
            value = bool_to_int(super()._print_Piecewise(expr))

        return value


class JuliaCodeGenerator(CodeGenerator):
    def __init__(self, ode: ODE, remove_unused: bool = False) -> None:
        super().__init__(ode, remove_unused=remove_unused)
        self._printer = GotranJuliaCodePrinter()
        # setattr(self, "_formatter", get_formatter(format=format))

    @property
    def printer(self):
        return self._printer

    @property
    def template(self):
        return templates.julia

    def imports(self) -> str:
        return ""
        # return self._format(
        #     "\n".join(
        #         [
        #             "#include <math.h>",
        #             "#include <string.h>\n",
        #         ]
        #     )
        # )

    def _rhs_arguments(
        self, order: RHSArgument | str = RHSArgument.stp, const_states: bool = True
    ) -> Func:
        value = RHSArgument.get_value(order)
        argument_dict = {
            "s": "states",
            "t": "t",
            "p": "parameters",
        }
        argument_list = [argument_dict[v] for v in value] + ["values"]
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,))
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,))
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,))

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            values_type="",
        )

    def _scheme_arguments(
        self,
        order: SchemeArgument | str = SchemeArgument.stdp,
        const_states: bool = True,
    ) -> Func:
        value = SchemeArgument.get_value(order)
        argument_dict = {
            "s": "states",
            "t": "t",
            "d": "dt",
            "p": "parameters",
        }
        argument_list = [argument_dict[v] for v in value] + ["values"]
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,))
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,))
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,))

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            values_type="",
        )
