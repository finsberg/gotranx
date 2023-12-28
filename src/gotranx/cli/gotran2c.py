from __future__ import annotations
from pathlib import Path
from structlog import get_logger

from ..codegen.c import CCodeGenerator
from ..load import load_ode

logger = get_logger()


def main(fname: Path, suffix: str = ".h", outname: str | None = None) -> None:
    ode = load_ode(fname)
    codegen = CCodeGenerator(ode)
    code = "\n".join(
        [
            "#include <math.h>",
            codegen.initial_parameter_values(),
            codegen.initial_state_values(),
            codegen.rhs(),
        ],
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(codegen._format(code))
    logger.info(f"Wrote {out_name}")
