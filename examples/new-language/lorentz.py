import jax
import jax.numpy as numpy

jax.config.update("jax_enable_x64", True)
parameter = {"beta": 0, "rho": 1, "sigma": 2}


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

    return parameter[name]


state = {"x": 0, "y": 1, "z": 2}


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

    return state[name]


monitor = {"dx_dt": 0, "dy_dt": 1, "dz_dt": 2}


def monitor_index(name: str) -> int:
    """Return the index of the monitor with the given name

    Arguments
    ---------
    name : str
        The name of the monitor

    Returns
    -------
    int
        The index of the monitor

    Raises
    ------
    KeyError
        If the name is not a valid monitor
    """

    return monitor[name]


@jax.jit
def init_parameter_values(**values):
    """Initialize parameter values"""
    # beta=2.4, rho=21.0, sigma=12.0

    parameters = numpy.array([2.4, 21.0, 12.0], dtype=numpy.float64)

    for key, value in values.items():
        parameters = parameters.at[parameter_index(key)].set(value)

    return parameters


@jax.jit
def init_state_values(**values):
    """Initialize state values"""
    # x=1.0, y=2.0, z=3.05

    states = numpy.array([1.0, 2.0, 3.05], dtype=numpy.float64)

    for key, value in values.items():
        states = states.at[state_index(key)].set(value)

    return states


@jax.jit
def rhs(t, states, parameters):

    # Assign states
    x = states[0]
    y = states[1]
    z = states[2]

    # Assign parameters
    beta = parameters[0]
    rho = parameters[1]
    sigma = parameters[2]

    # Assign expressions
    dx_dt = sigma * (-x + y)
    _values_0 = dx_dt
    dy_dt = x * (rho - z) - y
    _values_1 = dy_dt
    dz_dt = -beta * z + x * y
    _values_2 = dz_dt

    return numpy.array(
        [
            _values_0,
            _values_1,
            _values_2,
        ]
    )


@jax.jit
def monitor_values(t, states, parameters):

    # Assign states
    x = states[0]
    y = states[1]
    z = states[2]

    # Assign parameters
    beta = parameters[0]
    rho = parameters[1]
    sigma = parameters[2]

    # Assign expressions
    dx_dt = sigma * (-x + y)
    _values_0 = dx_dt
    dy_dt = x * (rho - z) - y
    _values_1 = dy_dt
    dz_dt = -beta * z + x * y
    _values_2 = dz_dt

    return numpy.array(
        [
            _values_0,
            _values_1,
            _values_2,
        ]
    )


@jax.jit
def forward_explicit_euler(states, t, dt, parameters):

    # Assign states
    x = states[0]
    y = states[1]
    z = states[2]

    # Assign parameters
    beta = parameters[0]
    rho = parameters[1]
    sigma = parameters[2]

    # Assign expressions
    dx_dt = sigma * (-x + y)
    _values_0 = dt * dx_dt + x
    dy_dt = x * (rho - z) - y
    _values_1 = dt * dy_dt + y
    dz_dt = -beta * z + x * y
    _values_2 = dt * dz_dt + z

    return numpy.array(
        [
            _values_0,
            _values_1,
            _values_2,
        ]
    )
