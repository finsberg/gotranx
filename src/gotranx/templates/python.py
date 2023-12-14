from textwrap import dedent, indent


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
    values_comment = ", ".join(
        [f"{n}={v}" for n, v in zip(state_names, state_values)],
    )
    values = ", ".join(map(str, state_values))
    return dedent(
        f'''
def init_state_values(**values):
    """Initialize state values
    """
    # {values_comment}

    {name} = np.array([{values}])

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
    values_comment = ", ".join(
        [f"{n}={v}" for n, v in zip(parameter_names, parameter_values)],
    )
    values = ", ".join(map(str, parameter_values))
    return dedent(
        f'''
def init_parameter_values(**values):
    """Initialize parameter values
    """
    # {values_comment}

    {name} = np.array([{values}])

    for key, value in values.items():
        {name}[parameter_index(key)] = value

    return {name}
''',
    )


def method(name, args, states, parameters, values, return_name: str):
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
{indent_values}

{indent_return}
""",
    )
