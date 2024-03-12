import gotranx
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

# Load the model
ode = gotranx.load_ode("ORdmm_Land.ode")

mechanics_comp = ode.get_component("mechanics")
mechanics_ode = mechanics_comp.to_ode()

ep_ode = ode - mechanics_comp

# Generate code and generalized rush larsen scheme
code = gotranx.cli.gotran2py.get_code(
    ode,
    scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
)

code_ep = gotranx.cli.gotran2py.get_code(
    ep_ode,
    scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
    missing_values=mechanics_ode.missing_variables,
)
code_mechanics = gotranx.cli.gotran2py.get_code(
    mechanics_ode,
    scheme=[gotranx.schemes.Scheme.forward_generalized_rush_larsen],
    missing_values=ep_ode.missing_variables,
)

# # Execute code
model: dict[str, Any] = {}
exec(code, model)
ep_model: dict[str, Any] = {}
exec(code_ep, ep_model)
mechanics_model: dict[str, Any] = {}
exec(code_mechanics, mechanics_model)


# Set time step to 0.1 ms
dt = 0.1
# Simulate model for 1000 ms
t = np.arange(0, 1000, dt)
# Get the index of the membrane potential
V_index_ep = ep_model["state_index"]("v")
Istim_index_ep = ep_model["monitor_index"]("Istim")


fgr_ep = ep_model["forward_generalized_rush_larsen"]
mon_ep = ep_model["monitor"]
mv_ep = ep_model["missing_values"]
Ca_index_ep = ep_model["state_index"]("cai")

fgr_mechanics = mechanics_model["forward_generalized_rush_larsen"]
mon_mechanics = mechanics_model["monitor"]
mv_mechanics = mechanics_model["missing_values"]

Ta_index_mechanics = mechanics_model["monitor_index"]("Ta")
J_TRPN_index_mechanics = mechanics_model["monitor_index"]("J_TRPN")

J_TRPN_index = model["monitor_index"]("J_TRPN")
Ta_index = model["monitor_index"]("Ta")
fgr = model["forward_generalized_rush_larsen"]
mon = model["monitor"]

tols = [2e-5, 4e-5, 6e-5, 8e-5, 1e-4]
colors = ["r", "g", "b", "c", "m"]

# Simulate the model
V_ep = np.zeros(len(t))
Ca_ep = np.zeros(len(t))

J_TRPN_full = np.zeros(len(t))
Ta_full = np.zeros(len(t))

Ta_mechanics = np.zeros(len(t))
J_TRPN_mechanics = np.zeros(len(t))

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 10))

for col, tol in zip(colors, tols):
    # Get initial state values
    y_ep = ep_model["init_state_values"]()
    p_ep = ep_model["init_parameter_values"]()
    ep_missing_index = ep_model["missing_index"]()
    ep_missing_values = np.zeros(len(ep_missing_index))

    y_mechanics = mechanics_model["init_state_values"]().astype(np.float64)
    p_mechanics = mechanics_model["init_parameter_values"]()
    mechanics_missing_index = mechanics_model["missing_index"]()
    mechanics_missing_values = np.zeros(len(mechanics_missing_index))

    y = model["init_state_values"]()
    p = model["init_parameter_values"]()

    # Ca_index_mechanics = mechanics_model["state_index"]("cai")
    # Get the default values of the missing values
    # A little bit chicken and egg problem here, but we can known that we don't need
    # the ep_missing_values is only the calcium concentration, which is a state variable
    mechanics_missing_values[:] = mv_ep(0, y_ep, p_ep, ep_missing_values)
    ep_missing_values[:] = mv_mechanics(0, y_mechanics, p_mechanics, mechanics_missing_values)

    prev_mechanics_missing_values = np.zeros_like(mechanics_missing_values)
    prev_mechanics_missing_values[:] = mechanics_missing_values

    inds = []
    for i, ti in enumerate(t):
        y = fgr(y, ti, dt, p)
        monitor = mon(ti, y, p)
        J_TRPN_full[i] = monitor[J_TRPN_index]
        Ta_full[i] = monitor[Ta_index]

        y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, ep_missing_values)
        V_ep[i] = y_ep[V_index_ep]
        Ca_ep[i] = y_ep[Ca_index_ep]
        monitor_ep = mon_ep(ti, y_ep, p_ep, ep_missing_values)

        mechanics_missing_values[:] = mv_ep(t, y_ep, p_ep, ep_missing_values)

        change = np.linalg.norm(
            mechanics_missing_values - prev_mechanics_missing_values
        ) / np.linalg.norm(prev_mechanics_missing_values)

        if change < tol:
            # Very small change to just continue to next time step
            continue

        inds.append(i)

        y_mechanics[:] = fgr_mechanics(y_mechanics, ti, dt, p_mechanics, mechanics_missing_values)
        monitor_mechanics = mon_mechanics(
            ti,
            y_mechanics,
            p_mechanics,
            mechanics_missing_values,
        )
        Ta_mechanics[i] = monitor_mechanics[Ta_index_mechanics]
        J_TRPN_mechanics[i] = monitor_mechanics[J_TRPN_index_mechanics]

        prev_mechanics_missing_values[:] = mechanics_missing_values
        ep_missing_values[:] = mv_mechanics(t, y_mechanics, p_mechanics, mechanics_missing_values)

    # Plot the results
    print(f"Solved on {100 * len(inds) / len(t)}% of the time steps")
    inds = np.array(inds)

    ax[0, 0].plot(t, V_ep, color=col)

    ax[1, 0].plot(t, Ta_full, color="k", linestyle="--")
    ax[1, 0].plot(t[inds], Ta_mechanics[inds], color=col)

    ax[0, 1].plot(t, Ca_ep, color=col)

    ax[1, 1].plot(t, J_TRPN_full, color="k", linestyle="--")
    ax[1, 1].plot(t[inds], J_TRPN_mechanics[inds], color=col)

    err_Ta = np.linalg.norm(Ta_full[inds] - Ta_mechanics[inds]) / np.linalg.norm(Ta_mechanics)
    err_J_TRPN = np.linalg.norm(J_TRPN_full[inds] - J_TRPN_mechanics[inds]) / np.linalg.norm(
        J_TRPN_mechanics
    )
    ax[2, 0].plot(
        t[inds],
        Ta_full[inds] - Ta_mechanics[inds],
        label=f"err={err_Ta:.2e}, tol={tol}",
        color=col,
    )

    ax[2, 1].plot(
        t[inds],
        J_TRPN_full[inds] - J_TRPN_mechanics[inds],
        label=f"err={err_J_TRPN:.2e}, tol={tol}",
        color=col,
    )

    ax[1, 0].set_xlabel("Time (ms)")
    ax[1, 1].set_xlabel("Time (ms)")
    ax[0, 0].set_ylabel("V (mV)")
    ax[1, 0].set_ylabel("Ta (kPa)")
    ax[0, 1].set_ylabel("Ca (mM)")
    ax[1, 1].set_ylabel("J TRPN (mM)")

    ax[2, 0].set_ylabel("Ta error (kPa)")
    ax[2, 1].set_ylabel("J TRPN error (mM)")
    ax[2, 0].legend()
    ax[2, 1].legend()

    fig.tight_layout()
    fig.savefig("V_and_Ta.png")
