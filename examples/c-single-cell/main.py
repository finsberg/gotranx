# # Solving a single cell in C
#
# In this demo we generate C code for a cardiac cell model, and solve a single cell with a
# small, standalone C program. This is how you would use the generated code from a solver
# written in C. If you instead want to call the generated C functions from Python, see
# [Compiling a C-extension](../compile-c-extension/main.py).
#
# We use the [ToR-ORd model](https://elifesciences.org/articles/48890) of a human ventricular
# cardiomyocyte. You can download the model in `.ode` format
# {download}`here <./ToRORd_dyn_chloride.ode>`.
#
# First we import what we need

import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import gotranx

# Generated files go in a separate `build` directory

here = Path.cwd()
build = here / "build"
build.mkdir(exist_ok=True)

# ## Generating the C code
#
# Next we load the model and generate C code for it, including the
# [generalized Rush-Larsen scheme](../../docs/schemes/generalized_rush_larsen.md) that we
# will use to step the cell forward in time

ode = gotranx.load_ode(here / "ToRORd_dyn_chloride.ode")
code = gotranx.cli.gotran2c.get_code(
    ode, scheme=[gotranx.schemes.Scheme.generalized_rush_larsen]
)
(build / "model.h").write_text(code)

# This is the same as running the following command on the command line
#
# ```
# gotranx ode2c ToRORd_dyn_chloride.ode --scheme generalized_rush_larsen -o build/model.h
# ```
#
# The header contains the number of states, parameters and monitored values, functions that
# set the initial values, functions that look up the index of a state, parameter or monitored
# value by name, and the functions `rhs`, `monitor_values` and `generalized_rush_larsen`. For
# example, the signature of the scheme is

signature = re.search(r"void generalized_rush_larsen\([^)]*\)", code).group()
print(" ".join(signature.split()))

# ## The C program
#
# The program `main.c` (shown in full at the bottom of this page) includes the generated
# header, sets the initial states and parameters, and steps the cell for two beats (2000 ms)
# with a time step of 0.01 ms. Every 0.1 ms it writes the time, the membrane potential `v`, the
# intracellular calcium concentration `cai` and the fast sodium current `INa` (a monitored
# value) to a CSV file. It looks these up by name with `state_index` and `monitor_index`, so it
# does not depend on the order of the states.
#
# We compile the program with the C compiler on the system

compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
if compiler is None:
    raise RuntimeError("This demo needs a C compiler (cc, gcc or clang)")
executable = build / "main"
subprocess.run(
    [compiler, "-O2", "-I", str(build), str(here / "main.c"), "-o", str(executable), "-lm"],
    check=True,
)

# and run it

results = build / "results.csv"
subprocess.run([str(executable), str(results)], check=True)

# ## The results
#
# Finally we load the CSV file and plot the results

data = np.loadtxt(results, delimiter=",", skiprows=1)
t, v, cai, INa = data.T

fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 8))
ax[0].plot(t, v)
ax[0].set_ylabel("v (mV)")
ax[1].plot(t, cai * 1000)
ax[1].set_ylabel("cai (µM)")
ax[2].plot(t, INa)
ax[2].set_ylabel("INa (µA/µF)")
ax[2].set_xlabel("Time (ms)")
fig.tight_layout()
plt.show()

# ## Source code
#
# ### `main.c`
# ```{literalinclude} ./main.c
# :language: c
# ```
