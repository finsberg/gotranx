"""Templates for the CUDA code generator.

Every function of the model becomes a kernel (`qualifier`) with one thread per cell: it
takes the number of cells as its last argument (`cell_count_argument`) and starts by
computing its cell and returning if that is past the end (`prologue`). The parameters and
the index lookups are host functions, the same as in C++.
"""

from __future__ import annotations
from textwrap import dedent, indent

from . import cpp
from .cpp import missing_index, monitor_index, parameter_index, state_index

__all__ = [
    "init_parameter_values",
    "init_state_values",
    "method",
    "missing_index",
    "monitor_index",
    "parameter_index",
    "state_index",
]


def init_parameter_values(
    name, parameter_names, parameter_values, code, float_type="double", **kwargs
):
    # A host function: the kernel keyword arguments do not apply
    return cpp.init_parameter_values(
        name, parameter_names, parameter_values, code, float_type=float_type
    )


def init_state_values(
    name,
    state_names,
    state_values,
    code,
    float_type="double",
    qualifier="__global__",
    prologue="",
    cell_count_argument="",
    **kwargs,
):
    indented_code = indent(code, "    ")
    indented_prologue = indent(prologue, "    ")
    indent_values = indent(", ".join(f"{n}={v}" for n, v in zip(state_names, state_values)), "    ")
    return dedent(
        f"""
{qualifier} void init_state_values({float_type}* {name}, {cell_count_argument}){{
    /*
{indent_values}
    */
{indented_prologue}
{indented_code}
}}
""",
    )


def method(
    name: str,
    args: str,
    states: str,
    parameters: str,
    values: str,
    return_name: None = None,
    num_return_values: int = 0,
    missing_variables: str = "",
    qualifier: str = "__global__",
    prologue: str = "",
    cell_count_argument: str = "",
    **kwargs,
):
    indent_prologue = indent(prologue, "    ")
    indent_states = indent(states, "    ")
    indent_parameters = indent(parameters, "    ")
    indent_values = indent(values, "    ")
    indent_missing_variables = indent(missing_variables, "    ")
    return dedent(
        f"""
{qualifier} void {name}({args}, {cell_count_argument}){{
{indent_prologue}

    // Assign states
{indent_states}

    // Assign parameters
{indent_parameters}
{indent_missing_variables}
    // Assign expressions
{indent_values}
}}
""",
    )
