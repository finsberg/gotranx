from __future__ import annotations
from pathlib import Path

from ..codegen import CCodeGenerator
from ..load import load_ode


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
    out.with_suffix(suffix=suffix).write_text(code)
