from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.python import PythonCodeGenerator, get_formatter
from ..load import load_ode
from ..schemes import Scheme
from ..ode import ODE

from .utils import add_schemes

logger = structlog.get_logger()


def get_code(
    ode: ODE,
    scheme: list[Scheme] | None = None,
    format: bool = True,
    remove_unused: bool = False,
    missing_values: dict[str, int] | None = None,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
) -> str:
    """Generate the Python code for the ODE

    Parameters
    ----------
    ode : gotranx.ode.ODE
        The ODE
    scheme : list[Scheme] | None, optional
        Optional numerical scheme, by default None
    format : bool, optional
        Apply ruff / black formatter, by default True
    remove_unused : bool, optional
        Remove unused variables, by default False
    missing_values : dict[str, int] | None, optional
        Missing values, by default None
    delta : float, optional
        Delta value for the rush larsen schemes, by default 1e-8
    stiff_states : list[str] | None, optional
        Stiff states, by default None. Only applicable for
        the hybrid rush larsen scheme

    Returns
    -------
    str
        The Python code
    """
    codegen = PythonCodeGenerator(
        ode,
        format=False,
        remove_unused=remove_unused,
    )
    formatter = get_formatter()
    if missing_values is not None:
        _missing_values = codegen.missing_values(missing_values)
    else:
        _missing_values = ""

    comp = [
        codegen.imports(),
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.missing_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
        _missing_values,
    ] + add_schemes(
        codegen,
        scheme=scheme,
        delta=delta,
        stiff_states=stiff_states,
    )
    code = codegen._format("\n".join(comp))
    if format:
        code = formatter(code)
    return code


def main(
    fname: Path,
    suffix: str = ".py",
    outname: str | None = None,
    format: bool = True,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    verbose: bool = True,
    stiff_states: list[str] | None = None,
    delta: float = 1e-8,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )

    ode = load_ode(fname)
    code = get_code(
        ode,
        scheme=scheme,
        format=format,
        remove_unused=remove_unused,
        delta=delta,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
