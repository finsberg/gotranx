# # Second-order Rush-Larsen schemes
#
# The Rush-Larsen family integrates the stiff gating variables of a cell model
# with an exponential update. The schemes `rush_larsen`,
# `generalized_rush_larsen` and `hybrid_rush_larsen` are first order: halving
# the time step halves the error. Their second-order counterparts
# `rush_larsen_2`, `generalized_rush_larsen_2` and `hybrid_rush_larsen_2`
# take a half step to a midpoint, evaluate the model there, and use that to
# take the full step. Halving the time step then quarters the error. This
# matters when the cell model is coupled to a tissue model with Strang
# splitting, which is only second order if the cell-model step is too.
#
# In this tutorial we use the
# [Beeler-Reuter model](https://doi.org/10.1113/jphysiol.1977.sp011853),
# which you can download {download}`here <./beeler_reuter_1977.ode>`.

import time as timer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

import gotranx
from gotranx.linearization import gating_states
from gotranx.schemes import Scheme

ode = gotranx.load_ode("beeler_reuter_1977.ode")

# The three families differ only in which states get the exponential update:
#
# - `generalized_rush_larsen{,_2}`: every state,
# - `rush_larsen{,_2}`: the *gating* states, i.e. those whose right-hand side
#   is affine in the state itself, detected automatically,
# - `hybrid_rush_larsen{,_2}`: the states you list in `stiff_states`.
#
# For Beeler-Reuter the gating states are the six gates:

print(gating_states(ode))

# Generate Python code for one first-order and two second-order schemes

schemes = [
    Scheme.rush_larsen,
    Scheme.generalized_rush_larsen,
    Scheme.rush_larsen_2,
    Scheme.generalized_rush_larsen_2,
]
code = gotranx.cli.gotran2py.get_code(ode, scheme=schemes)
model: dict[str, Any] = {}
exec(code, model)

y0 = model["init_state_values"]()
p = model["init_parameter_values"]()
T = 10.0  # ms: covers the stimulus (1-3 ms) and the upstroke

# A reference solution with a tight-tolerance implicit solver. The stimulus is
# switched on at 1 ms and off at 3 ms, so we integrate each piece separately.

y = np.array(y0, dtype=float)
for a, b in [(0.0, 1.0), (1.0, 3.0), (3.0, T)]:
    y = solve_ivp(
        lambda t, y: model["rhs"](t, y, p), (a, b), y, method="Radau", rtol=1e-12, atol=1e-12
    ).y[:, -1]
reference = y
scale = np.maximum(np.abs(reference), 1e-3)

# Now step each scheme to `T` with time steps $\Delta t = 2^{-k}$ ms, so that
# the stimulus edges fall on step boundaries (otherwise the jump in the
# stimulus, not the scheme, would limit the order), and record the error and
# the wall time.

dts = [2.0**-k for k in range(2, 9)]
results: dict[str, tuple[list[float], list[float]]] = {}
for scheme in schemes:
    step = model[scheme.value]
    errors, times = [], []
    for dt in dts:
        y, t = np.array(y0, dtype=float), 0.0
        start = timer.perf_counter()
        for _ in range(round(T / dt)):
            y = step(y, t, dt, p)
            t += dt
        times.append(timer.perf_counter() - start)
        errors.append(float(np.max(np.abs(y - reference) / scale)))
    results[scheme.value] = (errors, times)
    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    print(f"{scheme.value:28s} observed orders {np.round(rates, 2)}")

# The first-order schemes converge with order one and the second-order schemes
# approach order two as $\Delta t$ decreases. The asymptotic order only shows
# once the time step resolves the upstroke; at the coarsest steps the
# second-order schemes are still more accurate, but their observed order is
# lower.

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for label, (errors, times) in results.items():
    ax[0].loglog(dts, errors, "o-", label=label)
    ax[1].loglog(times, errors, "o-", label=label)
ax[0].loglog(dts, [errors[0] * (dt / dts[0]) for dt in dts], "k--", label="order 1")
ax[0].loglog(dts, [errors[0] * (dt / dts[0]) ** 2 for dt in dts], "k:", label="order 2")
ax[0].set_xlabel(r"$\Delta t$ (ms)")
ax[0].set_ylabel("relative error at T")
ax[1].set_xlabel("wall time (s)")
ax[1].set_ylabel("relative error at T")
ax[0].legend()
fig.tight_layout()
fig.savefig("second_order_rush_larsen.png")

# A second-order step costs roughly twice as much as a first-order one, since
# the model is evaluated twice. The right panel shows the trade-off: for a
# given accuracy, the second-order schemes are faster once the required
# accuracy is moderately high.
#
# 