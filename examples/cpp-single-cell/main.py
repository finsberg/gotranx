# # Two models in one C++ program
#
# In this demo we generate C++ code for two different cardiac cell models and solve a single
# cell of each, side by side, in one C++ program. This is something the
# [C code](../c-single-cell/main.py) cannot do: every model's C code defines functions with the
# same names (`rhs`, `init_state_values`, ...) and global constants (`NUM_STATES`, ...), so two
# models cannot be linked into one program. The C++ code instead puts each model in its own
# namespace, and defines its functions `inline` and its sizes `constexpr`, so a header can also
# be included from several source files.
#
# We use the [ToR-ORd model](https://elifesciences.org/articles/48890)
# ({download}`ToRORd_dyn_chloride.ode <./ToRORd_dyn_chloride.ode>`) and the
# [ten Tusscher-Panfilov model](https://journals.physiology.org/doi/full/10.1152/ajpheart.00109.2006)
# ({download}`tentusscher_panfilov_2006_M_cell.ode <./tentusscher_panfilov_2006_M_cell.ode>`).
#
# First we import what we need

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

# ## Generating the C++ code
#
# We generate one header per model, each in a namespace we choose, with the
# [generalized Rush-Larsen scheme](../../docs/schemes/generalized_rush_larsen.md)

models = {
    "tor_ord": "ToRORd_dyn_chloride.ode",
    "ten_tusscher": "tentusscher_panfilov_2006_M_cell.ode",
}
for namespace, odefile in models.items():
    ode = gotranx.load_ode(here / odefile)
    code = gotranx.cli.gotran2cpp.get_code(
        ode,
        scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
        namespace=namespace,
    )
    (build / f"{namespace}.hpp").write_text(code)

# This is the same as running
#
# ```
# gotranx ode2cpp ToRORd_dyn_chloride.ode --scheme generalized_rush_larsen --namespace tor_ord -o build/tor_ord.hpp
# gotranx ode2cpp tentusscher_panfilov_2006_M_cell.ode --scheme generalized_rush_larsen --namespace ten_tusscher -o build/ten_tusscher.hpp
# ```
#
# Without `--namespace`, the namespace is the name of the ODE (here `ToRORd_dyn_chloride` and
# `tentusscher_panfilov_2006_M_cell`). Each header starts like this

print("\n".join((build / "tor_ord.hpp").read_text().splitlines()[:10]))

# ## The C++ program
#
# The program `main.cpp` (shown in full at the bottom of this page) includes both headers. It
# sets up each model through its namespace (`tor_ord::init_state_values`,
# `ten_tusscher::NUM_STATES`, ...), steps both cells for one beat (1000 ms) with a time step of
# 0.01 ms, and every 0.1 ms writes the membrane potential of each model to a CSV file. The
# potential is called `v` in one model and `V` in the other, and each model's own
# `state_index` finds it.
#
# We compile the program with the C++ compiler on the system

compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
if compiler is None:
    raise RuntimeError("This demo needs a C++ compiler (c++, g++ or clang++)")
executable = build / "main"
subprocess.run(
    [compiler, "-std=c++17", "-O2", "-I", str(build), str(here / "main.cpp"), "-o", str(executable)],
    check=True,
)

# and run it

results = build / "results.csv"
subprocess.run([str(executable), str(results)], check=True)

# ## The results
#
# Finally we plot the action potential of both models

data = np.loadtxt(results, delimiter=",", skiprows=1)
t, v_tor_ord, v_ten_tusscher = data.T

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(t, v_tor_ord, label="ToR-ORd")
ax.plot(t, v_ten_tusscher, label="ten Tusscher-Panfilov")
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Membrane potential (mV)")
ax.legend()
fig.tight_layout()
plt.show()

# ## Source code
#
# ### `main.cpp`
# ```{literalinclude} ./main.cpp
# :language: c++
# ```
