from __future__ import annotations
from textwrap import dedent, indent
import functools


def acc(all_values: str, next_value: str = "# ") -> str:
    if len(all_values.split("\n")[-1]) + len(next_value) > 60:
        return all_values + "\n#" + next_value
    else:
        return all_values + ", " + next_value


def state_index(data: dict[str, int]) -> str:
    return dedent(
        f'''
def state_index(name: str) -> float:
    """Return the index of the state with the given name

    Arguments
    ---------
    name : str
        The name of the state

    Returns
    -------
    float
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
    return dedent(
        f'''
def parameter_index(name: str) -> float:
    """Return the index of the parameter with the given name

    Arguments
    ---------
    name : str
        The name of the parameter

    Returns
    -------
    float
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


def method(name, args, states, parameters, values, return_name: str, num_return_values: int):
    indent_states = indent(states, "    ")
    indent_parameters = indent(parameters, "    ")
    indent_values = indent(values, "    ")
    indent_return = indent("return " + return_name, "    ")
    return dedent(
        f"""
def {name}({args}):

    # Assign states
{indent_states}

    # Assign parameters
{indent_parameters}

    # Assign expressions
    {return_name} = numpy.zeros({num_return_values})
{indent_values}

{indent_return}
""",
    )
