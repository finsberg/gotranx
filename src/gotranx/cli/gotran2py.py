from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.python import PythonCodeGenerator
from ..load import load_ode
from ..schemes import Scheme
from ..ode import ODE

logger = structlog.get_logger()


def get_code(
    ode: ODE,
    scheme: list[Scheme] | None = None,
    apply_black: bool = True,
    remove_unused: bool = False,
    missing_values: dict[str, int] | None = None,
) -> str:
    """Generate the Python code for the ODE

    Parameters
    ----------
    ode : ODE
        The ODE
    scheme : list[Scheme] | None, optional
        Optional numerical scheme, by default None
    apply_black : bool, optional
        Apply black formatter, by default True
    remove_unused : bool, optional
        Remove unused variables, by default False

    Returns
    -------
    str
        The Python code
    """
    codegen = PythonCodeGenerator(
        ode,
        apply_black=apply_black,
        remove_unused=remove_unused,
    )
    if missing_values is not None:
        _missing_values = codegen.missing_values(missing_values)
    else:
        _missing_values = ""

    comp = [
        "import math",
        "import numpy",
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.missing_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
        _missing_values,
    ]

    if scheme is not None:
        for s in scheme:
            comp.append(codegen.scheme(s.value))

    return codegen._format("\n".join(comp))


def main(
    fname: Path,
    suffix: str = ".py",
    outname: str | None = None,
    apply_black: bool = True,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    verbose: bool = True,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )

    ode = load_ode(fname)
    code = get_code(
        ode,
        scheme=scheme,
        apply_black=apply_black,
        remove_unused=remove_unused,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
