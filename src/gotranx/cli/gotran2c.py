from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.c import CCodeGenerator, Format, get_formatter
from ..load import load_ode
from ..schemes import Scheme
from ..ode import ODE

from .utils import add_schemes

logger = structlog.get_logger()


def get_code(
    ode: ODE,
    scheme: list[Scheme] | None = None,
    format: Format = Format.clang_format,
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
    format : gotranx.codegen.python.Format, optional
        The formatter, by default gotranx.codegen.python.Format.black
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
        The C code
    """
    codegen = CCodeGenerator(ode, remove_unused=remove_unused, format=Format.none)
    formatter = get_formatter(format=format)

    if missing_values is not None:
        _missing_values = codegen.missing_values(missing_values)
    else:
        _missing_values = ""

    comp = [
        codegen.imports(),
        f"int NUM_STATES = {len(ode.states)};",
        f"int NUM_PARAMS = {len(ode.parameters)};",
        f"int NUM_MONITORED = {len(ode.state_derivatives) + len(ode.intermediates)};",
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

    if format != Format.none:
        logger.debug("Applying formatter", format=format)
        code = formatter(code)

    return code


def main(
    fname: Path,
    suffix: str = ".h",
    outname: str | None = None,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    format: Format = Format.clang_format,
    verbose: bool = False,
    missing_values: dict[str, int] | None = None,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
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
        missing_values=missing_values,
        delta=delta,
        stiff_states=stiff_states,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
