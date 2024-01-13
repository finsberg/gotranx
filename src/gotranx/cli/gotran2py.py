from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.python import PythonCodeGenerator
from ..load import load_ode
from ..schemes import Scheme

logger = structlog.get_logger()


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
    codegen = PythonCodeGenerator(ode, apply_black=apply_black, remove_unused=remove_unused)
    comp = [
        "import math",
        "import numpy",
        codegen.parameter_index(),
        codegen.state_index(),
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
