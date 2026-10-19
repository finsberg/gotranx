from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.c import Format
from ..codegen.cpp import FloatType
from ..codegen.cuda import CUDACodeGenerator
from ..load import load_ode
from ..schemes import Scheme
from ..ode import ODE

from .gotran2cpp import assemble

logger = structlog.get_logger()


def get_code(
    ode: ODE,
    scheme: list[Scheme] | None = None,
    format: Format = Format.clang_format,
    remove_unused: bool = False,
    missing_values: dict[str, int] | None = None,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
    cse: bool = True,
    namespace: str | None = None,
    float_type: FloatType | str = FloatType.double,
) -> str:
    """Generate the CUDA code for the ODE

    Parameters
    ----------
    ode : gotranx.ode.ODE
        The ODE
    scheme : list[Scheme] | None, optional
        Optional numerical scheme, by default None
    format : gotranx.codegen.c.Format, optional
        The formatter, by default gotranx.codegen.c.Format.clang_format
    remove_unused : bool, optional
        Remove unused variables, by default False
    missing_values : dict[str, int] | None, optional
        Missing values, by default None
    delta : float, optional
        Delta value for the rush larsen schemes, by default 1e-8
    stiff_states : list[str] | None, optional
        Stiff states, by default None. Only applicable for
        the hybrid rush larsen scheme
    cse : bool, optional
        Factor out subexpressions shared by the linearized expressions in
        the rush larsen schemes, by default True. False inlines them
        instead, emitting no temporaries
    namespace : str | None, optional
        The namespace of the generated code, by default the name of the ODE
    float_type : FloatType | str, optional
        The floating point type, "double" (default) or "float"

    Returns
    -------
    str
        The CUDA code: a header with a kernel per function, in `namespace <namespace>`
    """
    codegen = CUDACodeGenerator(
        ode,
        remove_unused=remove_unused,
        format=Format.none,
        namespace=namespace,
        float_type=float_type,
    )
    return assemble(
        codegen,
        scheme=scheme,
        format=format,
        missing_values=missing_values,
        delta=delta,
        stiff_states=stiff_states,
        cse=cse,
    )


def main(
    fname: Path,
    suffix: str = ".cuh",
    outname: Path | str | None = None,
    scheme: list[Scheme] | None = None,
    remove_unused: bool = False,
    remove_singularities: bool = True,
    format: Format = Format.clang_format,
    verbose: bool = False,
    missing_values: dict[str, int] | None = None,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
    cse: bool = True,
    namespace: str | None = None,
    float_type: FloatType | str = FloatType.double,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )
    ode = load_ode(fname, remove_singularities=remove_singularities)
    code = get_code(
        ode,
        scheme=scheme,
        format=format,
        remove_unused=remove_unused,
        missing_values=missing_values,
        delta=delta,
        stiff_states=stiff_states,
        cse=cse,
        namespace=namespace,
        float_type=float_type,
    )
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=suffix)
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
