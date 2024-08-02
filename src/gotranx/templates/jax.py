from __future__ import annotations
from textwrap import dedent, indent
import functools
from structlog import get_logger

from .python import acc, state_index, parameter_index, monitor_index, missing_index

logger = get_logger()


def init_state_values(name, state_names, state_values, code):
    logger.debug(f"Generating init_state_values with {len(state_values)} values")
    values_comment = indent(
        "#" + functools.reduce(acc, [f"{n}={v}" for n, v in zip(state_names, state_values)]),
        "    ",
    )

    values = ", ".join(map(str, state_values))
    return dedent(
        f'''
@jax.jit
def init_state_values(**values):
    """Initialize state values
    """
{values_comment}

    {name} = numpy.array([{values}], dtype=numpy.float64)

    for key, value in values.items():
        {name} = {name}.at[state_index(key)].set(value)

    return {name}
''',
    )


def init_parameter_values(name, parameter_names, parameter_values, code):
    logger.debug(f"Generating init_parameter_values with {len(parameter_values)} values")
    values_comment = indent(
        "#"
        + functools.reduce(acc, [f"{n}={v}" for n, v in zip(parameter_names, parameter_values)]),
        "    ",
    )

    values = ", ".join(map(str, parameter_values))
    return dedent(
        f'''
@jax.jit
def init_parameter_values(**values):
    """Initialize parameter values
    """
{values_comment}

    {name} = numpy.array([{values}], dtype=numpy.float64)

    for key, value in values.items():
        {name} = {name}.at[parameter_index(key)].set(value)

    return {name}
''',
    )


def method(
    name,
    args,
    states,
    parameters,
    values,
    num_return_values: int,
    missing_variables: str = "",
    **kwargs,
):
    logger.debug(f"Generating method '{name}', with {num_return_values} return values.")
    if len(kwargs) > 0:
        logger.debug(f"Unused kwargs: {kwargs}")

    return_name_lst = (
        ["numpy.array(["] + [f"_values_{i}, " for i in range(num_return_values)] + ["])"]
    )
    indent_return = indent(f"return {''.join(return_name_lst)}", "    ")
    indent_missing_variables = indent(missing_variables, "    ")
    indent_states = indent(states, "    ")
    indent_parameters = indent(parameters, "    ")
    indent_values = indent(values, "    ")

    return dedent(
        f"""
@jax.jit
def {name}({args}):

    # Assign states
{indent_states}

    # Assign parameters
{indent_parameters}
{indent_missing_variables}
    # Assign expressions
{indent_values}

{indent_return}
""",
    )


__all__ = [
    "init_state_values",
    "init_parameter_values",
    "method",
    "parameter_index",
    "state_index",
    "monitor_index",
    "missing_index",
]
