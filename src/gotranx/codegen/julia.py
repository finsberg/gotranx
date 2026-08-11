from __future__ import annotations
import structlog
from sympy.printing.julia import JuliaCodePrinter
from sympy.codegen.ast import Assignment
import sympy

from ..ode import ODE
from .. import templates
from .base import CodeGenerator, Func, RHSArgument, SchemeArgument

logger = structlog.get_logger()


def bool_to_int(expr: str) -> str:
    return expr.replace("false", "0").replace("true", "1")


class GotranJuliaCodePrinter(JuliaCodePrinter):
    def __init__(self, type_stable: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._type_stable = type_stable
        self._settings["contract"] = False

    def _print_Float(self, flt):
        value = str(float(flt))
        if self._type_stable:
            return self._print(f"TYPE({value})")
        return self._print(value)

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

    def _print_Indexed(self, expr):
        inds = [self._print(i + 1) for i in expr.indices]  # Reindex arrays to start at 1
        return "%s[%s]" % (self._print(expr.base.label), ",".join(inds))


class JuliaCodeGenerator(CodeGenerator):
    def __init__(self, ode: ODE, remove_unused: bool = False, type_stable: bool = False) -> None:
        super().__init__(ode, remove_unused=remove_unused)
        self._printer = GotranJuliaCodePrinter(type_stable=type_stable)
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
        if self._printer._type_stable:
            argument_dict = {
                "s": "states::AbstractVector{TYPE}",
                "t": "t::TYPE",
                "p": "parameters::AbstractVector{TYPE}",
            }
            values = ["values::AbstractVector{TYPE}"]
            post_function_signature = " where {TYPE}"
        else:
            argument_dict = {
                "s": "states",
                "t": "t",
                "p": "parameters",
            }
            values = ["values"]
            post_function_signature = ""
        argument_list = [argument_dict[v] for v in value] + values
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,), offset=1)
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,), offset=1)
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,), offset=1)

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            values_type="",
            post_function_signature=post_function_signature,
        )

    def _scheme_arguments(
        self,
        order: SchemeArgument | str = SchemeArgument.stdp,
        const_states: bool = True,
    ) -> Func:
        value = SchemeArgument.get_value(order)
        if self._printer._type_stable:
            argument_dict = {
                "s": "states::AbstractVector{TYPE}",
                "t": "t::TYPE",
                "d": "dt::TYPE",
                "p": "parameters::AbstractVector{TYPE}",
            }
            values = ["values::AbstractVector{TYPE}"]
            post_function_signature = " where {TYPE}"
        else:
            argument_dict = {
                "s": "states",
                "t": "t",
                "d": "dt",
                "p": "parameters",
            }
            values = ["values"]
            post_function_signature = ""

        argument_list = [argument_dict[v] for v in value] + values
        states = sympy.IndexedBase("states", shape=(self.ode.num_states,), offset=1)
        parameters = sympy.IndexedBase("parameters", shape=(self.ode.num_parameters,), offset=1)
        values = sympy.IndexedBase("values", shape=(self.ode.num_states,), offset=1)

        return Func(
            arguments=argument_list,
            states=states,
            parameters=parameters,
            values=values,
            values_type="",
            post_function_signature=post_function_signature,
        )
