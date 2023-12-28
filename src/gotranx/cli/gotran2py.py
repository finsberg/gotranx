from __future__ import annotations
from pathlib import Path

from ..codegen.python import PythonCodeGenerator, Backend
from ..load import load_ode


def main(
    fname: Path,
    suffix: str = ".h",
    outname: str | None = None,
    apply_black: bool = True,
    backend: Backend = Backend.numpy,
) -> None:
    ode = load_ode(fname)
    codegen = PythonCodeGenerator(ode, apply_black=apply_black, backend=backend)
    code = "\n".join(
        [
            "import math",
            "import numpy",
            codegen.parameter_index(),
            codegen.state_index(),
            codegen.initial_parameter_values(),
            codegen.initial_state_values(),
            codegen.rhs(),
            codegen.scheme("forward_euler"),
            codegen.scheme("forward_generalized_rush_larsen"),
        ],
    )
    out = fname if outname is None else Path(outname)
    out.with_suffix(suffix=suffix).write_text(codegen._format(code))
