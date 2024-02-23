import matplotlib.pyplot as plt
import utils

model = utils.load_model("ORdmm_Land.ode")

y = model.initial_state_values()
# Get initial parameter values
p = model.initial_parameter_values()
parameters = model.parameter_values_to_dict(p)

# Simulate the model
sol = model.solve(0, 1000, dt=0.01, u0=y, parameters=parameters)

V = sol["v"]
Ca = sol["cai"]

m = sol.monitor(["Ta", "Istim"])
Ta = m[:, 0]
Istim = m[:, 1]

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
fig.savefig("V_and_Ta_c.png")
