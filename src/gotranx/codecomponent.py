from __future__ import annotations
import typing
import attr
from .ode import ODE


@attr.s(frozen=True, slots=True)
class CodeComponent:
    name: str = attr.ib()
    ode: ODE = attr.ib()
    function_name: str = attr.ib()
    description: str = attr.ib()
    params: dict[str, typing.Any] | None = attr.ib(default=None)


def rhs_expressions(ode: ODE, function_name: str = "rhs", result_name: str = "dy", params=None):
    descr = f"Compute the right hand side of the {ode} ODE"
    return CodeComponent(
        name="RHSComponent",
        ode=ode,
        function_name=function_name,
        description=descr,
        params=params,
        **{result_name: ode.state_derivatives},
    )
