from __future__ import annotations
from pathlib import Path
import logging
import enum
import structlog

from ..codegen.base import Shape
from ..codegen.jax import JaxCodeGenerator
from ..codegen.python import PythonCodeGenerator, get_formatter, Format
from ..load import load_ode
from ..schemes import Scheme
from ..ode import ODE

from .utils import add_schemes

logger = structlog.get_logger()


class Backend(str, enum.Enum):
    numpy = "numpy"
    jax = "jax"


def get_code(
    ode: ODE,
    scheme: list[Scheme] | None = None,
    format: Format = Format.black,
    remove_unused: bool = False,
    missing_values: dict[str, int] | None = None,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
    backend: Backend = Backend.numpy,
    shape: Shape = Shape.dynamic,
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
    backend : Backend, optional
        The backend, by default Backend.numpy
    shape : Shape, optional
        The shape of the output arrays, by default Shape.dynamic


    Returns
    -------
    str
        The Python code
    """
    if backend == Backend.numpy:
        CodeGenerator = PythonCodeGenerator
    elif backend == Backend.jax:
        CodeGenerator = JaxCodeGenerator
    else:
        raise ValueError(f"Unknown backend {backend}")

    codegen = CodeGenerator(
        ode,
        format=Format.none,
        remove_unused=remove_unused,
        shape=shape,
    )
    formatter = get_formatter(format=format)
    if missing_values is not None:
        _missing_values = codegen.missing_values(missing_values)
    else:
        _missing_values = ""

    comp = [
        codegen.imports(),
        codegen.parameter_index(),
        codegen.state_index(),
        codegen.monitor_index(),
        codegen.missing_index(),
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
        _missing_values,
    ] + add_schemes(
        codegen,
        scheme=scheme,
        delta=delta,
        stiff_states=stiff_states,
    )
    code = codegen._format("\n".join(comp))

    if format != Format.none:
        # Run the formatter only once
        logger.debug("Applying formatter", format=format)
        code = formatter(code)
    return code


def main(
    fname: Path,
    outname: str | None = None,
    format: Format = Format.black,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    verbose: bool = True,
    stiff_states: list[str] | None = None,
    delta: float = 1e-8,
    suffix: str = ".py",
    backend: Backend = Backend.numpy,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )

    ode = load_ode(fname)

    code = get_code(
        ode,
        scheme=scheme,
        format=format,
        remove_unused=remove_unused,
        stiff_states=stiff_states,
        delta=delta,
        backend=backend,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
