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
    cse: bool = True,
    precompute_constants: bool = False,
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
    cse : bool, optional
        Factor out subexpressions shared by the linearized expressions in
        the rush larsen schemes, by default True. False inlines them
        instead, emitting no temporaries
    precompute_constants : bool, optional
        Compute every intermediate that depends only on parameters once, in a generated
        `init_constants(parameters, constants)`, and read it from the `constants` array,
        which `rhs`, `monitor_values` and the schemes then take right after
        `parameters`, by default False. Call `init_constants` again after changing a
        parameter

    Returns
    -------
    str
        The C code
    """
    codegen = CCodeGenerator(
        ode,
        remove_unused=remove_unused,
        format=Format.none,
        precompute_constants=precompute_constants,
    )
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
        *([f"int NUM_CONSTANTS = {len(codegen.constant_index)};"] if precompute_constants else []),
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.missing_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        *([codegen.init_constants()] if precompute_constants else []),
        codegen.rhs(),
        codegen.monitor_values(),
        _missing_values,
    ] + add_schemes(
        codegen,
        scheme=scheme,
        delta=delta,
        stiff_states=stiff_states,
        cse=cse,
    )

    code = codegen._format("\n".join(comp))

    if format != Format.none:
        logger.debug("Applying formatter", format=format)
        code = formatter(code)

    return code


def main(
    fname: Path,
    suffix: str = ".h",
    outname: Path | str | None = None,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    remove_singularities: bool = True,
    format: Format = Format.clang_format,
    verbose: bool = False,
    missing_values: dict[str, int] | None = None,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
    cse: bool = True,
    precompute_constants: bool = False,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )
    ode = load_ode(fname, remove_singularities=remove_singularities)
    code = get_code(
        ode,
        scheme=scheme,
        format=format,
        remove_unused=remove_unused,
        missing_values=missing_values,
        delta=delta,
        stiff_states=stiff_states,
        cse=cse,
        precompute_constants=precompute_constants,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
