from __future__ import annotations
from pathlib import Path

from ..codegen.python import PythonCodeGenerator
from ..load import load_ode


def main(fname: Path, suffix: str = ".h", outname: str | None = None) -> None:
    ode = load_ode(fname)
    codegen = PythonCodeGenerator(ode)
    code = "\n".join(
        [
            "import math",
            "import numpy as np",
            codegen.parameter_index(),
            codegen.state_index(),
            codegen.initial_parameter_values(),
            codegen.initial_state_values(),
            codegen.rhs(),
        ],
    )
    out = fname if outname is None else Path(outname)
    out.with_suffix(suffix=suffix).write_text(codegen._format(code))
