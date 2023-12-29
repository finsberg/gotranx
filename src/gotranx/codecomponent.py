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
    """Generate the right hand side of the ODE

    Parameters
    ----------
    ode : ODE
        The ODE
    function_name : str, optional
        The name of the function, by default "rhs"
    result_name : str, optional
        The name of the result, by default "dy"
    params : dict[str, typing.Any], optional
        Additional parameters to pass to the template, by default None

    Returns
    -------
    CodeComponent
        The code component
    """

    descr = f"Compute the right hand side of the {ode} ODE"
    return CodeComponent(
        name="RHSComponent",
        ode=ode,
        function_name=function_name,
        description=descr,
        params=params,
        **{result_name: ode.state_derivatives},
    )
