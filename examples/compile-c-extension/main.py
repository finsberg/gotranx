#  # Compiling a C-extension
#
# In this demo we will show how to set up a system that takes your `.ode` file, generates C-code, compiles the code just-in-time and imports the functions into python again. Note that there are several steps involved in this process we therefore split the code across two different modules `utils.py` and `cmodel.py`
#
# To start with we will just run through an example how how this can be used.
#
# First we import the modules `utils` where we add all the functionality as well as `matplotlib` for plotting
#
# ```{note}
# The full source code for the all the files need for this demo (including `utils.py`) is found at the bottom of this document
# ```
#

import matplotlib.pyplot as plt
import utils


# For this tutorial we will use a rather large system of ODE which simulated the electromechanics in cardiac cells that are based on the [O'Hara-Rudy model for electrophysiology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002061) and the [Land model](https://www.sciencedirect.com/science/article/abs/pii/S0022282817300639). You can download the model in `.ode` format {download}`here <./ORdmm_Land.ode>`
#
# Next we load the model. This is function contains the functionality for generating code and compiling the C-extension. Currently it will also regenerate the code as well as recompiling the code code every time you run the code. It is also possible to only do this if the relevant files do not exist.

model = utils.load_model("ORdmm_Land.ode", rebuild=True, regenerate=True)

# Next we get the initial states and parameters. The parameters we get back from `initial_parameter_values` are a `numpy` array. We also convert the parameters to a dictionary which is easier to work with since we can use the name of the parameter rather than the index.

y = model.initial_state_values()
# Get initial parameter values
p = model.initial_parameter_values()
parameters = model.parameter_values_to_dict(p)

# Next we solve the model for 1000.0 milliseconds with a time step of `0.01` ms. Note that it would also be possible to make this loop in python (similar to [the python API demo](../run-python/main.py)), however we will get a lot of performance gain if we instead do this loop in C.

# Simulate the model
sol = model.solve(0, 1000, dt=0.01, u0=y, parameters=parameters)

# Now let us extract the state variables for the voltage and intracellular calcium

V = sol["v"]
Ca = sol["cai"]

# as well as some monitor values

m = sol.monitor(["Ta", "Istim"])
Ta = m[:, 0]
Istim = m[:, 1]

# and finally we plot the results

# Plot the results
fig, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].plot(sol.time, V)
ax[1, 0].plot(sol.time, Ta)
ax[0, 1].plot(sol.time, Ca)
ax[1, 1].plot(sol.time, Istim)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel("Istim (uA/cm^2)")
fig.tight_layout()
plt.show()

# ## Source code
#
# ### `utils.py`
# ```{literalinclude} ./utils.py
# ```
#
# ### `cmodel.py`
# ```{literalinclude} ./cmodel.py
# ```
#
# ### `template.c`
# ```{literalinclude} ./template.c
# ```
#
# ### `CMakeLists.txt`
# ```{literalinclude} ./CMakeLists.txt
# ```
