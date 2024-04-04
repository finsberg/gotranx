from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.c import CCodeGenerator
from ..load import load_ode
from ..schemes import Scheme
from ..ode import ODE

logger = structlog.get_logger()


def get_code(
    ode: ODE,
    scheme: list[Scheme] | None = None,
    apply_clang_format: bool = True,
    remove_unused: bool = False,
) -> str:
    """Generate the Python code for the ODE

    Parameters
    ----------
    ode : ODE
        The ODE
    scheme : list[Scheme] | None, optional
        Optional numerical scheme, by default None
    apply_clang_format : bool, optional
        Apply clang formatter, by default True
    remove_unused : bool, optional
        Remove unused variables, by default False

    Returns
    -------
    str
        The C code
    """
    codegen = CCodeGenerator(
        ode, remove_unused=remove_unused, apply_clang_format=apply_clang_format
    )
    comp = [
        "#include <math.h>",
        "#include <string.h>\n",
        f"int NUM_STATES = {len(ode.states)};",
        f"int NUM_PARAMS = {len(ode.parameters)};",
        f"int NUM_MONITORED = { len(ode.state_derivatives) + len(ode.intermediates)};",
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
    ]
    if scheme is not None:
        for s in scheme:
            comp.append(codegen.scheme(s.value))

    return codegen._format("\n".join(comp))


def main(
    fname: Path,
    suffix: str = ".h",
    outname: str | None = None,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    apply_clang_format: bool = True,
    verbose: bool = False,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )
    ode = load_ode(fname)
    code = get_code(
        ode,
        scheme=scheme,
        apply_clang_format=apply_clang_format,
        remove_unused=remove_unused,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
