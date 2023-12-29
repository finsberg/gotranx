[![pre-commit](https://github.com/finsberg/gotranx/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/finsberg/gotranx/actions/workflows/pre-commit.yml)
[![CI](https://github.com/finsberg/gotranx/actions/workflows/main.yml/badge.svg)](https://github.com/finsberg/gotranx/actions/workflows/main.yml)
[![github pages](https://github.com/finsberg/gotranx/actions/workflows/pages.yml/badge.svg)](https://finsberg.github.io/gotranx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# gotranx

`gotranx` is the next generation General ODE translator and is an attempt to a full rewrite of [gotran](https://github.com/ComputationalPhysiology/gotran).

The general idea is that you write your ODE in a high level markup language and use `gotranx` to generate code for solving the ODE in different programming languages.

`gotranx` uses [`sympy`](https://www.sympy.org/en/index.html) to generate the numerical schemes.

**Note: `gotranx` is still under active development, so do not expect it to be complete.**

## Install
Install with pip
```
python3 -m pip install gotranx
```
or for the development version
```
python3 -m pip install git+https://github.com/finsberg/gotranx
```

## Quick start
Define you ode in an `.ode` file, for example for the Lorentz attractor you can write the following file (called `lorentz.ode`)
```
parameters(
sigma=12.0,
rho=21.0,
beta=2.4
)

states(x=1.0, y=2.0,z=3.05)

dx_dt = sigma * (y - x)
dy_dt = x * (rho - z) - y
dz_dt = x * y - beta * z
```
which defines the parameters and states with default initial conditions. For each state-variable we also define the derivatives using `dx_dt` for the state with name `x`.

Now we can generate code that can be used to solve the equation, e.g
```
python3 -m gotranx lorentz.ode --to .py --scheme forward_explicit_euler
```
which will generate a python file `lorentz.py` containing the following functions
```python
import numpy

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

    data = {"beta": 0, "rho": 1, "sigma": 2}
    return data[name]


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

    data = {"x": 0, "y": 1, "z": 2}
    return data[name]


def init_parameter_values(**values):
    """Initialize parameter values"""
    # beta=2.4, rho=21.0, sigma=12.0

    parameters = numpy.array([2.4, 21.0, 12.0])

    for key, value in values.items():
        parameters[parameter_index(key)] = value

    return parameters


def init_state_values(**values):
    """Initialize state values"""
    # x=1.0, y=2.0, z=3.05

    states = numpy.array([1.0, 2.0, 3.05])

    for key, value in values.items():
        states[state_index(key)] = value

    return states


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
    values = numpy.zeros(3)
    values[0] = sigma * (-x + y)
    values[1] = x * (rho - z) - y
    values[2] = -beta * z + x * y

    return values


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
    values = numpy.zeros(3)
    dx_dt = sigma * (-x + y)
    values[0] = dt * dx_dt + x
    dy_dt = x * (rho - z) - y
    values[1] = dt * dy_dt + y
    dz_dt = -beta * z + x * y
    values[2] = dt * dz_dt + z

    return values

```

Similarly, you can use the following command
```
python3 -m gotranx lorentz.ode --to .h --scheme forward_explicit_euler
```
to generate a C-header file with the following content
```C
#include <math.h>

void init_parameter_values(double *parameters)
{
    /*
    beta=2.4, rho=21.0, sigma=12.0
    */
    parameters[0] = 2.4;
    parameters[1] = 21.0;
    parameters[2] = 12.0;
}

void init_state_values(double *states)
{
    /*
    x=1.0, y=2.0, z=3.05
    */
    states[0] = 1.0;
    states[1] = 2.0;
    states[2] = 3.05;
}

void rhs(const double t, const double *__restrict states, const double *__restrict parameters, double *values)
{

    // Assign states
    const double x = states[0];
    const double y = states[1];
    const double z = states[2];

    // Assign parameters
    const double beta = parameters[0];
    const double rho = parameters[1];
    const double sigma = parameters[2];

    // Assign expressions
    values[0] = sigma * (-x + y);
    values[1] = x * (rho - z) - y;
    values[2] = -beta * z + x * y;
}

void forward_explicit_euler(const double *__restrict states, const double t, const double dt,
                            const double *__restrict parameters, double *values)
{

    // Assign states
    const double x = states[0];
    const double y = states[1];
    const double z = states[2];

    // Assign parameters
    const double beta = parameters[0];
    const double rho = parameters[1];
    const double sigma = parameters[2];

    // Assign expressions
    const double dx_dt = sigma * (-x + y);
    values[0] = dt * dx_dt + x;
    const double dy_dt = x * (rho - z) - y;
    values[1] = dt * dy_dt + y;
    const double dz_dt = -beta * z + x * y;
    values[2] = dt * dz_dt + z;
}
```
