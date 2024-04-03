from __future__ import annotations
from textwrap import dedent, indent
import functools
from structlog import get_logger

logger = get_logger()


def acc(all_values: str, next_value: str = "# ") -> str:
    if len(all_values.split("\n")[-1]) + len(next_value) > 60:
        return all_values + "\n#" + next_value
    else:
        return all_values + ", " + next_value


def _index(data: dict[str, int], name: str) -> str:
    return dedent(
        f'''
{name} = {repr(data)}


def {name}_index(name: str) -> int:
    """Return the index of the {name} with the given name

    Arguments
    ---------
    name : str
        The name of the {name}

    Returns
    -------
    int
        The index of the {name}

    Raises
    ------
    KeyError
        If the name is not a valid {name}
    """

    return {name}[name]
''',
    )


def state_index(data: dict[str, int]) -> str:
    logger.debug(f"Generating state_index with {len(data)} values")
    return _index(data, "state")


def parameter_index(data: dict[str, int]) -> str:
    logger.debug(f"Generating parameter_index with {len(data)} values")
    return _index(data, "parameter")


def monitor_index(data: dict[str, int]) -> str:
    logger.debug(f"Generating monitored_index with {len(data)} values")
    return _index(data, "monitor")


def missing_index(data: dict[str, int]) -> str:
    logger.debug(f"Generating missing_index with {len(data)} values")
    return _index(data, "missing")


def init_state_values(name, state_names, state_values, code):
    logger.debug(f"Generating init_state_values with {len(state_values)} values")
    values_comment = indent(
        "#" + functools.reduce(acc, [f"{n}={v}" for n, v in zip(state_names, state_values)]),
        "    ",
    )

    values = ", ".join(map(str, state_values))
    return dedent(
        f'''
def init_state_values(**values):
    """Initialize state values
    """
{values_comment}

    {name} = numpy.array([{values}], dtype=numpy.float64)

    for key, value in values.items():
        {name}[state_index(key)] = value

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
def init_parameter_values(**values):
    """Initialize parameter values
    """
{values_comment}

    {name} = numpy.array([{values}], dtype=numpy.float64)

    for key, value in values.items():
        {name}[parameter_index(key)] = value

    return {name}
''',
    )


def method(
    name,
    args,
    states,
    parameters,
    values,
    return_name: str,
    num_return_values: int,
    nan_to_num: bool = False,
    values_type: str = "numpy.zeros_like(states, dtype=numpy.float64)",
    shape_info: str = "",
    missing_variables: str = "",
    **kwargs,
):
    logger.debug(f"Generating method '{name}', with {num_return_values} return values.")
    if len(kwargs) > 0:
        logger.debug(f"Unused kwargs: {kwargs}")

    indent_missing_variables = indent(missing_variables, "    ")
    indent_states = indent(states, "    ")
    indent_parameters = indent(parameters, "    ")
    indent_values = indent(values, "    ")
    if nan_to_num:
        indent_return = indent(
            f"return numpy.nan_to_num({return_name}, nan=0.0)",
            "    ",
        )
    else:
        indent_return = indent(f"return {return_name}", "    ")
    return dedent(
        f"""
def {name}({args}):

    # Assign states
{indent_states}

    # Assign parameters
{indent_parameters}
{indent_missing_variables}
    # Assign expressions
    {shape_info}
    {return_name} = {values_type}
{indent_values}

{indent_return}
""",
    )
