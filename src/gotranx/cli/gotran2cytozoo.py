from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.cytozoo import CytoZooCodeGenerator
from ..load import load_ode
from ..ode import ODE

logger = structlog.get_logger()


def get_code(
    ode: ODE,
    model_name: str | None = None,
    v_name: str = "v",
    remove_unused: bool = False,
) -> str:
    """Generate a CytoZoo cell-model adapter for the ODE."""
    codegen = CytoZooCodeGenerator(
        ode, model_name=model_name, v_name=v_name, remove_unused=remove_unused, type_stable=True
    )

    comp = [
        "import CytoZoo",
        "using CytoZoo: AbstractCardiacCellModel, SpatialContext, resolve_parameter",
        "using CytoZoo: num_states, num_parameters, transmembrane_potential_index",
        "using CytoZoo: default_initial_state, state_index, parameter_index",
        "using CytoZoo: state_names, parameter_names",
        "using CytoZoo: num_monitors, monitor_names, monitor_values!",
        f"const NUM_STATES = {len(ode.states)};",
        f"const NUM_PARAMS = {len(ode.parameters)};",
        codegen.initial_parameter_values(),
        codegen.initial_state_values(),
        codegen.rhs(),
        codegen.monitor_values(),
        codegen.adapter(),
    ]
    return codegen._format("\n".join(comp))


def main(
    fname: Path,
    outname: Path | str | None = None,
    model_name: str | None = None,
    v_name: str = "v",
    remove_unused: bool = False,
    remove_singularities: bool = True,
    verbose: bool = False,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(loglevel))
    ode = load_ode(fname, remove_singularities=remove_singularities)
    code = get_code(ode, model_name=model_name, v_name=v_name, remove_unused=remove_unused)
    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(suffix=".jl")
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
