# # Your first ODE file
# In this tutorial we will show how to write your own ODE file from scratch. We will use Lorentz system as an example. The Lorentz system is a system of ordinary differential equations first studied by Edward Lorenz. It is notable for having chaotic solutions for certain parameter values and initial conditions. In particular, the Lorenz attractor is a set of chaotic solutions of the Lorenz system which, when plotted, resemble a butterfly or figure eight.
# The equations are given by:
#
# ```{math}
# \begin{align}
# \frac{dx}{dt} &= \sigma(y - x) \\
# \frac{dy}{dt} &= x(\rho - z) - y \\
# \frac{dz}{dt} &= x y - \beta z
# \end{align}
# ```
# where $\sigma$, $\rho$ and $\beta$ are the parameters of the system.
#
# When solving the system we will use the following parameters:
#
# ```{math}
# \begin{align}
# \sigma &= 12 \\
# \rho &= 21 \\
# \beta &= 2.4
# \end{align}
# ```
#
# We will also use the following initial conditions:
#
# ```{math}
# \begin{align}
# x(0) &= 1.0 \\
# y(0) &= 2.0 \\
# z(0) &= 3.05
# \end{align}
# ```
#
# Let's start by writing the ODE file.

ode_str = """
# This is the Lorentz system
# And it is part of this tutorial

parameters(
sigma=12.0,
rho=21.0,
beta=2.4
)

states(x=ScalarParam(1.0, unit="m", description="x variable"), y=2.0,z=3.05)

dx_dt = sigma * (y - x)  # The derivative of x
dy_dt = x * (rho - z) - y  # m/s
dz_dt = x * y - beta * z
"""

# Note that we have also add a description and a unit to the state `x`, and some description and units to `dx_dt` and `dy_dt` respectively. We can now save the file to disk in a file called `lorentz.ode` and load it with `gotranx.load_ode("lorentz.ode")`, or we can just create the ODE from a string directly

# +
import gotranx

ode = gotranx.load.ode_from_string(ode_str)
# -

# Let us first print the ODE

print(ode)

# We can also print the states

from pprint import pprint

pprint(ode.states)

# and we see that we have three states. Similarly we can print the parameters

pprint(ode.parameters)

# and the state derivatives.

pprint(ode.state_derivatives)

# We could also print the intermediates, but there are no intermediate variables (i.e variables that are not states, parameters nor state derivatives) in the model

pprint(ode.intermediates)

# In the ODE file we also added some text at the top. This can be accessed using the `text` attribute

print(ode.text)

# We can also get a dictionary with all the sympy symbols using i the model

print(ode.symbols)

# Here the keys are the names of the variables and the values are the sympy symbols. We can also see which variables depend on which variables

print(ode.dependents())

# Here for example `dz_dt` depends only on `beta`, while all state derivatives depend on the state `y`. Now let use take a closer look at one of the states, for example `x` where

x = ode["x"]
print(x)

# Can print the value

print(x.value)

# which is actually a sympy object.

print(type(x.value))

# We can also print the unit

print(x.unit)

# which is a [pint](https://pint.readthedocs.io/en/stable/) unit object. The original string is stored in the `unit_str` attribute

print(x.unit_str)

# Similarly we can print the description

print(x.description)

# and the name and symbol

print(x.name, type(x.name))
print(x.symbol, type(x.symbol))

# For `dx_dt` we added a comment. This can be accessed using the `comment` attribute

dx_dt = ode["dx_dt"]
print(dx_dt.comment)


# Note also that the full tree used when parsing the expression is also saved in the value

print(dx_dt.value.tree.pretty())

# We can also print the dependencies of the expression

print(dx_dt.value.dependencies)

# For the state derivatives `dy_dt` we also added a comment, but this was meant to be a unit. When parsing the comment, `gotranx` first tries to parse the comment as a unit, and if that fails it will be stored as a comment. We can access the unit using the `unit` attribute

dy_dt = ode["dy_dt"]
print(dy_dt.unit)


# Now to generate code can create a code generator object

codegen = gotranx.codegen.PythonCodeGenerator(ode)

# and to generate code for the initial states for example we can just call the `initial_state_values` method

print(codegen.initial_state_values())

# To generate code for a specific scheme you can use the `scheme` method and pass the scheme and the order of the arguments, for example


print(codegen.scheme(f=gotranx.get_scheme("forward_explicit_euler"), order="stdp"))

# Note that with this order you get the arguments in the order  `states`, `time`, `dt` and `parameters`.  Passing `order="ptsd"` will give the following order

print(codegen.scheme(f=gotranx.get_scheme("forward_explicit_euler"), order="ptsd"))

# To list the available schemes you can do

print(gotranx.schemes.list_schemes())

# So let us try to solve it using the forward euler scheme. We can use the `gotranx.cli.gotran2py.get_code` function to generate the code for all the necessary functions

code = gotranx.cli.gotran2py.get_code(
    ode, scheme=[gotranx.schemes.Scheme.forward_explicit_euler]
)

# and then execute the code

from typing import Any

model: dict[str, Any] = {}
exec(code, model)

# We can now use the model to simulate the system

import numpy as np
import matplotlib.pyplot as plt

y = model["init_state_values"]()
p = model["init_parameter_values"]()
dt = 0.01
t = np.arange(0, 100, dt)
x_index = model["state_index"]("x")
y_index = model["state_index"]("y")
z_index = model["state_index"]("z")
fgr = model["forward_explicit_euler"]

state = np.zeros((3, len(t)))
for i, ti in enumerate(t):
    y = fgr(y, ti, dt, p)
    state[0, i] = y[x_index]
    state[1, i] = y[y_index]
    state[2, i] = y[z_index]

# And plot the results

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.plot(state[0, :], state[1, :], state[2, :], lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")
plt.show()
