from __future__ import annotations

from pathlib import Path
import logging
import structlog

from ..codegen.mtk import MTKCodeGenerator
from ..load import load_ode
from ..ode import ODE

logger = structlog.get_logger()


def get_code(
    ode: ODE,
    remove_unused: bool = False,
) -> str:
    """Generate ModelingToolkit.jl code for the ODE."""
    codegen = MTKCodeGenerator(ode, remove_unused=remove_unused)
    return codegen.generate()


def main(
    fname: Path,
    outname: str | None = None,
    remove_unused: bool = False,
    verbose: bool = False,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )
    ode = load_ode(fname)
    code = get_code(
        ode,
        remove_unused=remove_unused,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=".jl")
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
