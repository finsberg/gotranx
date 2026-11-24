# # Splitting an ODE into two sub-ODEs
#
# In some cases it might be useful to split an ODE into two separate ODEs, for example when you are modeling different dynamics and these are happening on different time scales. One example of this is when we model both the electrical and mechanical dynamics of heart cells. We can model them within the same ODE, but you might want to embed the model inside a 3D tissue model, in which it is important to solve the dependent variables within the correct model (the PDEs for mechanics are typically more expensive to solve, so we want to solve them less frequently)
#
# In this demo we will build a model of the electromechanics in cardiac cells from two existing models, and then split it again:
#
# 1. **Fetch** the [O'Hara-Rudy model for electrophysiology](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002061) (`ohara_rudy_2011`, version `pmr-e65b98c`, a CellML file) and the [Land model](https://www.sciencedirect.com/science/article/abs/pii/S0022282817300639) of contraction (`land_2017`, version `github-6e5a5aa`, an `.ode` file) from [odezoo](https://github.com/ComputationalPhysiology/odezoo), a registry of cardiac models.
# 2. **Compose** them with {py:func}`gotranx.compose`. A small glue file, `ord_land.glue.ode`, says how they are coupled: Land's troponin replaces the rapid troponin buffering of O'Hara-Rudy, the calcium that Land sees is that of O'Hara-Rudy, and the troponin flux of Land enters the calcium balance of O'Hara-Rudy (as in [Margara et al. (2021)](https://doi.org/10.1098/rsta.2019.0345)). Land is in seconds, µM and Pa, and O'Hara-Rudy in milliseconds and mM, and the glue converts between them. See the [composition how-to](../../docs/composition.md) for the recipe and the glue file.
# 3. **Split** the composed model with {py:meth}`gotranx.ODE.split` into a `mechanics` component, which needs the calcium concentration and gives the active tension and the troponin flux, and an electrophysiology remainder.
#
# The numbers differ somewhat from earlier versions of this example, which used `ORdmm`, a modified O'Hara-Rudy model: this is the original 2011 model, so the action potential (for instance its peak) and the calcium transient differ somewhat, and the active tension by a few percent.
#
# Then we solve the full model, and the two halves separately with the mechanics stepped less often.
#
# First let's do the necessary imports

from pathlib import Path
import gotranx
import odezoo
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

# Fetch the two models. `odezoo.get` downloads a model once and returns the path of the cached file.

ord_path = odezoo.get("ohara_rudy_2011", version="pmr-e65b98c")
land_path = odezoo.get("land_2017", version="github-6e5a5aa")

# The coupling is written in the glue file `ord_land.glue.ode`, next to this script. It uses the `.ode` syntax, plus three statements of its own: `ep.cai` is the name `cai` of the part `ep`, `replace` gives an existing parameter or expression a new definition, and `+=` adds a term to a state's derivative. A plain statement, such as `J_TRPN = ...`, defines a new name.

print(Path("ord_land.glue.ode").read_text())

# The recipe lists the parts and their time units, and points at the glue file (it is next to this script). Land has a parameter `a`, an intermediate `h` and an intermediate `dLambda`, which collide with names in O'Hara-Rudy or the glue, so we rename them. The recipe could equally be a TOML file, but here the paths come from odezoo so we write it as a dictionary.

recipe = {
    "time_unit": "ms",
    "glue": "ord_land.glue.ode",
    "parts": [
        {"name": "ep", "file": str(ord_path), "time_unit": "ms"},
        {
            "name": "land",
            "file": str(land_path),
            "time_unit": "s",
            "rename": {"a": "a_land", "h": "h_land", "dLambda": "dLambda_land"},
        },
    ],
}
ode = gotranx.compose(recipe, base_dir=Path.cwd())

# The composed model is an ordinary `gotranx.ODE`, with the components of the two parts as `ep.<component>` and `land.<component>`. We now split out a component called `"mechanics"`: it takes the calcium concentration `cai` as an input and the active tension `Ta` and the troponin flux `J_TRPN` are its outputs. Everything that is needed to compute them moves into the new component.

ode = ode.split("mechanics", inputs=["cai"], outputs=["Ta", "J_TRPN"])
mechanics_comp = ode.get_component("mechanics")
mechanics_ode = mechanics_comp.to_ode()

# We can now find the remaining ODE by subtracting the mechanics ODE from the full ODE

ep_ode = ode - mechanics_comp

# The variables that cross between the two are the missing variables of each side: the mechanics model needs the calcium concentration from EP, and EP needs the troponin flux from mechanics

print("Mechanics needs:", mechanics_ode.missing_variables)
print("EP needs:", ep_ode.missing_variables)

# Now let us generate code for all the ODEs. We generate code for the full model

code = gotranx.cli.gotran2py.get_code(
    ode,
    scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
)

# the electrophysiology model

code_ep = gotranx.cli.gotran2py.get_code(
    ep_ode,
    scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    missing_values=mechanics_ode.missing_variables,
)

# and for the mechanics model

code_mechanics = gotranx.cli.gotran2py.get_code(
    mechanics_ode,
    scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    missing_values=ep_ode.missing_variables,
)

# Now to actually get the models we can execute them into their own namespace (i.e dictionaries)

model: dict[str, Any] = {}
exec(code, model)
ep_model: dict[str, Any] = {}
exec(code_ep, ep_model)
mechanics_model: dict[str, Any] = {}
exec(code_mechanics, mechanics_model)

# The default parameters of Land are those of a skinned cell (one without a cell membrane), with a calcium sensitivity `ca50_ref` of 2.5 µM and a reference tension `Tref` of 40.5 kPa. We want an intact human cell, so we set the parameters to the values of the Land model as it is used with the O'Hara-Rudy model (in Land's units: rates in 1/s, tensions in Pa, times in s), and let the cross-bridges start fully blocked (`B = 1`). We pass the same values to the full model and to the mechanics half.

land_parameters = dict(
    k_trpn=100.0,
    ntrpn=2.0,
    trpn50=0.35,
    rw=0.5,
    rs=0.25,
    gs=8.5,
    gw=615.0,
    phi=2.23,
    Aeff=25.0,
    beta0=2.3,
    beta1=-2.4,
    ca50_ref=0.805,
    Tref=120_000.0,
    kuw=182.0,
    kws=12.0,
    ku=40.0,
    nTm=2.4,
    a_land=2100.0,
    b=9.1,
    k=7.0,
    eta_l=0.2,
    eta_s=0.02,
)
land_states = dict(B=1.0, CaTRPN=0.0)

# The stimulus of the CellML model starts at 50 ms (`stimStart`), and we start it at 0 ms. The mechanics half does not have that parameter, and neither does it have the passive parameters, so for each module we pick out the overrides that it has. We check that every name is a parameter of the full model, so that a typo is not silently dropped everywhere.

overrides = {**land_parameters, "stimStart": 0.0}
unknown = sorted(set(overrides) - set(model["parameter"]))
assert not unknown, f"Not parameters of the model: {unknown}"


def overrides_for(module: dict[str, Any]) -> dict[str, float]:
    return {k: v for k, v in overrides.items() if k in module["parameter"]}


# We set the time step to 0.1 ms, and simulate the model for 1000 ms

dt = 0.1
t = np.arange(0, 1000, dt)

# Now we need to set up all variables

# Get the index of the membrane potential
V_index_ep = ep_model["state_index"]("v")
# Forward generalized rush larsen scheme for the electrophysiology model
fgr_ep = ep_model["generalized_rush_larsen"]
# Monitor function for the electrophysiology model
mon_ep = ep_model["monitor_values"]
# Missing values function for the electrophysiology model
mv_ep = ep_model["missing_values"]
# Index of the calcium concentration
Ca_index_ep = ep_model["state_index"]("cai")
# Forward generalized rush larsen scheme for the mechanics model
fgr_mechanics = mechanics_model["generalized_rush_larsen"]
# Monitor function for the mechanics model
mon_mechanics = mechanics_model["monitor_values"]
# Missing values function for the mechanics model
mv_mechanics = mechanics_model["missing_values"]
# Index of the active tension
Ta_index_mechanics = mechanics_model["monitor_index"]("Ta")
# Index of the J_TRPN
J_TRPN_index_mechanics = mechanics_model["monitor_index"]("J_TRPN")
# Forward generalized rush larsen scheme for the full model
fgr = model["generalized_rush_larsen"]
# Monitor function for the full model
mon = model["monitor_values"]
# Index of the active tension for the full model
Ta_index = model["monitor_index"]("Ta")
# Index of the J_TRPN for the full model
J_TRPN_index = model["monitor_index"]("J_TRPN")

# We will now see how many steps we need to ensure that the missing variables that are passed from the EP model to the mechanics model are below a certain tolerance. Lower tolerance will typically mean more steps, and we would like to see how the solution depends on how often we solve the ODE

# Tolerances to test for when to perform steps in the mechanics model
tols = [2e-5, 4e-5, 6e-5]
# Colors for the plots
colors = ["r", "g", "b", "c", "m"]

# We create some arrays to store the results

# +
# Create arrays to store the results
V_ep = np.zeros(len(t))
Ca_ep = np.zeros(len(t))
J_TRPN_full = np.zeros(len(t))
Ta_full = np.zeros(len(t))

Ta_mechanics = np.zeros(len(t))
J_TRPN_mechanics = np.zeros(len(t))
# -

# And we run the loop

fig, ax = plt.subplots(3, 2, sharex=True, figsize=(10, 10))
for j, (col, tol) in enumerate(zip(colors, tols)):
    # Get initial values from the EP model
    y_ep = ep_model["init_state_values"]()
    p_ep = ep_model["init_parameter_values"](**overrides_for(ep_model))
    ep_missing_values = np.zeros(len(ep_ode.missing_variables))

    # Get initial values from the mechanics model
    y_mechanics = mechanics_model["init_state_values"](**land_states)
    p_mechanics = mechanics_model["init_parameter_values"](**overrides_for(mechanics_model))
    mechanics_missing_values = np.zeros(len(mechanics_ode.missing_variables))

    # Get the initial values from the full model
    y = model["init_state_values"](**land_states)
    p = model["init_parameter_values"](**overrides_for(model))

    # Get the default values of the missing values
    # A little bit chicken and egg problem here, but in this specific case we know that the mechanics_missing_values is only the calcium concentration, which is a state variable and this doesn't require any additional information to be calculated.
    mechanics_missing_values[:] = mv_ep(0, y_ep, p_ep, ep_missing_values)
    ep_missing_values[:] = mv_mechanics(0, y_mechanics, p_mechanics, mechanics_missing_values)

    # We will store the previous missing values to check for convergence
    prev_mechanics_missing_values = np.zeros_like(mechanics_missing_values)
    prev_mechanics_missing_values[:] = mechanics_missing_values

    inds = []
    for i, ti in enumerate(t):
        # Forward step for the full model
        y[:] = fgr(y, ti, dt, p)
        monitor = mon(ti, y, p)
        J_TRPN_full[i] = monitor[J_TRPN_index]
        Ta_full[i] = monitor[Ta_index]

        # Forward step for the EP model
        y_ep[:] = fgr_ep(y_ep, ti, dt, p_ep, ep_missing_values)
        V_ep[i] = y_ep[V_index_ep]
        Ca_ep[i] = y_ep[Ca_index_ep]
        monitor_ep = mon_ep(ti, y_ep, p_ep, ep_missing_values)

        # Update missing values for the mechanics model
        mechanics_missing_values[:] = mv_ep(ti, y_ep, p_ep, ep_missing_values)

        # Compute the change in the missing values
        change = np.linalg.norm(
            mechanics_missing_values - prev_mechanics_missing_values
        ) / np.linalg.norm(prev_mechanics_missing_values)

        # Check if the change is small enough to continue to the next time step
        if change < tol:
            # Very small change to just continue to next time step
            continue

        # Store the index of the time step where we performed a step
        inds.append(i)

        # Forward step for the mechanics model
        y_mechanics[:] = fgr_mechanics(y_mechanics, ti, dt, p_mechanics, mechanics_missing_values)
        monitor_mechanics = mon_mechanics(
            ti,
            y_mechanics,
            p_mechanics,
            mechanics_missing_values,
        )
        Ta_mechanics[i] = monitor_mechanics[Ta_index_mechanics]
        J_TRPN_mechanics[i] = monitor_mechanics[J_TRPN_index_mechanics]

        # Update missing values for the EP model
        ep_missing_values[:] = mv_mechanics(t, y_mechanics, p_mechanics, mechanics_missing_values)

        prev_mechanics_missing_values[:] = mechanics_missing_values

    # Plot the results
    print(f"Solved on {100 * len(inds) / len(t)}% of the time steps")
    inds = np.array(inds)

    ax[0, 0].plot(t, V_ep, color=col, label=f"tol={tol}")

    if j == 0:
        # Plot the full model with a dashed line only for the first run
        ax[1, 0].plot(t, Ta_full, color="k", linestyle="--", label="Full")
        ax[1, 1].plot(t, J_TRPN_full, color="k", linestyle="--", label="Full")

    ax[1, 0].plot(t[inds], Ta_mechanics[inds], color=col, label=f"tol={tol}")

    ax[0, 1].plot(t, Ca_ep, color=col, label=f"tol={tol}")

    ax[1, 1].plot(t[inds], J_TRPN_mechanics[inds], color=col, label=f"tol={tol}")

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

    for axi in ax.flatten():
        axi.legend()

    fig.tight_layout()
plt.show()
