"""Python template for code generation"""

from __future__ import annotations
from textwrap import dedent, indent
import functools
from structlog import get_logger

logger = get_logger()


def acc(all_values: str, next_value: str = "# ") -> str:
    """Accumulate values in a string

    Arguments
    ---------
    all_values : str
        The accumulated values
    next_value : str
        The next value to add

    Returns
    -------
    str
        The accumulated values
    """

    if len(all_values.split("\n")[-1]) + len(next_value) > 60:
        return all_values + "\n#" + next_value
    else:
        return all_values + ", " + next_value


def _index(data: dict[str, int], name: str) -> str:
    """Helper function to generate index function for a given data

    Arguments
    ---------
    data : dict[str, int]
        The data
    name : str
        The name of the function

    Returns
    -------
    str
        The index function
    """
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
    """The state_index function is a function that returns
    the index of the state with the given name.

    Parameters
    ----------
    data : dict[str, int]
        The data containing the state names and their indexes

    Returns
    -------
    str
        The code for the state_index function
    """

    logger.debug(f"Generating state_index with {len(data)} values")
    return _index(data, "state")


def parameter_index(data: dict[str, int]) -> str:
    """The parameter_index function is a function that returns
    the index of the parameter with the given name.

    Parameters
    ----------
    data : dict[str, int]
        The data containing the parameter names and their indexes

    Returns
    -------
    str
        The code for the parameter_index function
    """

    logger.debug(f"Generating parameter_index with {len(data)} values")
    return _index(data, "parameter")


def monitor_index(data: dict[str, int]) -> str:
    """The monitor_index function is a function that returns
    the index of the monitor with the given name.

    A monitor is a variable that is not part of the state or parameter
    but is used to monitor the simulation. For example, in a cardiac
    cell models, different currents across the membrane are examples
    of monitors.

    Parameters
    ----------
    data : dict[str, int]
        The data containing the monitor names and their indexes

    Returns
    -------
    str
        The code for the monitor_index function
    """
    logger.debug(f"Generating monitored_index with {len(data)} values")
    return _index(data, "monitor")


def missing_index(data: dict[str, int]) -> str:
    """The missing_index function is a function that returns
    the index of the missing value with the given name.

    A missing value is a variable that is not part of the state or parameter
    but is used in the right-hand side of the ODE. This typically happens
    when the model is not fully defined, and some variables are missing, for
    example when you split an ODE into two sub-models and variables needs
    to be passed between them.

    Parameters
    ----------
    data : dict[str, int]
        The data containing the missing value names and their indexes

    Returns
    -------
    str
        The code for the missing_index function
    """
    logger.debug(f"Generating missing_index with {len(data)} values")
    return _index(data, "missing")


def init_state_values(name, state_names, state_values, code):
    """The init_state_values function is a function that initializes
    the state values of the model.

    The function typically return an array of the state values, or in
    the case of C code, it takes a pointer to the array of state values
    and initializes the values in place.

    Parameters
    ----------
    name : str
        Name of the return variable
    state_values : list[float]
        Default values for the states
    state_names : list[str]
        The names of the states
    code : str
        Additional code to be inserted into the function (not used in this template)

    Returns
    -------
    str
        The code for the init_state_values function
    """
    logger.debug(f"Generating init_state_values with {len(state_values)} values")
    if len(state_values) == 0:
        values_comment = ""
    else:
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
    """The init_parameter_values function is a function that initializes
    the parameter values of the model.

    The function typically return an array of the parameter values, or in
    the case of C code, it takes a pointer to the array of parameter values
    and initializes the values in place.

    Parameters
    ----------
    name : str
        Name of the return variable
    parameter_values : list[float]
        Default values for the parameters
    parameter_names : list[str]
        The names of the parameters
    code : str
        Additional code to be inserted into the function (not used in this template)

    Returns
    -------
    str
        The code for the init_parameter_values function
    """
    logger.debug(f"Generating init_parameter_values with {len(parameter_values)} values")
    if len(parameter_values) == 0:
        values_comment = ""
    else:
        values_comment = indent(
            "#"
            + functools.reduce(
                acc,
                [f"{n}={v}" for n, v in zip(parameter_names, parameter_values)],
            ),
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
    """The method function is a function that generates a method
    for the model.

    One example of a method is the right-hand side of the ODE, which
    calculates the derivatives of the states. Another example is the
    monitor values, which calculates the values of the monitors.
    All the different numerical schemes are also examples of methods.


    Parameters
    ----------
    name : str
        The name of the method
    args : str
        The arguments of the method
    states : str
        The code for assigning the states
    parameters : str
        The code for assigning the parameters
    values : str
        The code for assigning the values
    return_name : str | None
        The name of the return variable
    num_return_values : int
        The number of return values
    shape_info : str
        The shape information of the return values
    values_type : str
        The type of the values
    missing_variables : str
        The code for handling missing variables
    nan_to_num : bool
        If True, the return values are passed through numpy.nan_to_num
        with nan=0.0

    Returns
    -------
    str
        The code for the method
    """

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
