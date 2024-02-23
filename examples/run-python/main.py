import gotranx
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

# Load the model
ode = gotranx.load_ode("ORdmm_Land.ode")
# Generate code and generalized rush larsen scheme
code = gotranx.cli.gotran2py.get_code(
    ode, scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen]
)
# Execute code
model: dict[str, Any] = {}
exec(code, model)

# Get initial state values
y = model["init_state_values"]()  # **init_state_values("Torord_endo_BARS_1000"))
# Get initial parameter values
p = model["init_parameter_values"]()
# Set time step to 0.1 ms
dt = 0.1
# Simulate model for 1000 ms
t = np.arange(0, 1000, dt)
# Get the index of the membrane potential
V_index = model["state_index"]("v")
Ca_index = model["state_index"]("cai")
# Get the index of the active tension from the land model
Ta_index = model["monitor_index"]("Ta")
Istim_index = model["monitor_index"]("Istim")

fgr = model["forward_generalized_rush_larsen"]
mon = model["monitor"]

# Simulate the model
V = np.zeros(len(t))
Ca = np.zeros(len(t))
Ta = np.zeros(len(t))
Istim = np.zeros(len(t))
for i, ti in enumerate(t):
    y = fgr(y, ti, dt, p)
    V[i] = y[V_index]
    Ca[i] = y[Ca_index]
    monitor = mon(ti, y, p)
    Ta[i] = monitor[Ta_index]
    Istim[i] = monitor[Istim_index]

# Plot the results
fig, ax = plt.subplots(2, 2, sharex=True)
ax[0, 0].plot(t, V)
ax[1, 0].plot(t, Ta)
ax[0, 1].plot(t, Ca)
ax[1, 1].plot(t, Istim)
ax[1, 0].set_xlabel("Time (ms)")
ax[1, 1].set_xlabel("Time (ms)")
ax[0, 0].set_ylabel("V (mV)")
ax[1, 0].set_ylabel("Ta (kPa)")
ax[0, 1].set_ylabel("Ca (mM)")
ax[1, 1].set_ylabel("Istim (uA/cm^2)")
fig.tight_layout()
fig.savefig("V_and_Ta.png")
