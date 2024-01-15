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


def state_index(data: dict[str, int]) -> str:
    logger.debug(f"Generating state_index with {len(data)} values")
    return dedent(
        f'''
def state_index(name: str) -> int:
    """Return the index of the state with the given name

    Arguments
    ---------
    name : str
        The name of the state

    Returns
    -------
    int
        The index of the state

    Raises
    ------
    KeyError
        If the name is not a valid state
    """

    data = {repr(data)}
    return data[name]
''',
    )


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

    {name} = numpy.array([{values}])

    for key, value in values.items():
        {name}[state_index(key)] = value

    return {name}
''',
    )


def parameter_index(data: dict[str, int]) -> str:
    logger.debug(f"Generating parameter_index with {len(data)} values")
    return dedent(
        f'''
def parameter_index(name: str) -> int:
    """Return the index of the parameter with the given name

    Arguments
    ---------
    name : str
        The name of the parameter

    Returns
    -------
    int
        The index of the parameter

    Raises
    ------
    KeyError
        If the name is not a valid parameter
    """

    data = {repr(data)}
    return data[name]
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

    {name} = numpy.array([{values}])

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
    **kwargs,
):
    logger.debug(f"Generating method '{name}', with {num_return_values} return values.")
    if len(kwargs) > 0:
        logger.debug(f"Unused kwargs: {kwargs}")
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

    # Assign expressions
    {return_name} = numpy.zeros_like(states)
{indent_values}

{indent_return}
""",
    )
