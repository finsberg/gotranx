from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.c import CCodeGenerator
from ..load import load_ode
from ..schemes import Scheme

logger = structlog.get_logger()


def main(
    fname: Path,
    suffix: str = ".h",
    outname: str | None = None,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    verbose: bool = False,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )
    ode = load_ode(fname)
    codegen = CCodeGenerator(ode, remove_unused=remove_unused)
    comp = [
        "#include <math.h>",
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
    ]
    if scheme is not None:
        for s in scheme:
            comp.append(codegen.scheme(s.value))

    code = "\n".join(comp)
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(codegen._format(code))
    logger.info(f"Wrote {out_name}")
