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

# Generate Python code for two first-order and two second-order schemes

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

# The first-order schemes converge with order one. The second-order schemes
# approach order two as $\Delta t$ decreases, reaching 1.9 or more at the finest
# steps, but the asymptotic order only shows once the time step resolves the
# upstroke. The errors at the coarsest and finest time steps show the
# consequence: at $\Delta t = 0.25$ ms all four schemes have errors of about
# 10 %, and the second-order schemes are not clearly better (for
# `rush_larsen_2` the error is even slightly larger than for `rush_larsen`),
# whereas at $\Delta t = 2^{-8}$ ms they are more than ten times smaller than
# the first-order ones.

print("relative error at T, dt = 0.25 ms and dt = 2^-8 ms")
for name, (errors, _) in results.items():
    print(f"{name:28s} {errors[0]:.2e} {errors[-1]:.2e}")

# Cost per step at the finest time step, and the cost of each second-order
# scheme relative to its first-order counterpart:

steps = round(T / dts[-1])
cost = {name: times[-1] / steps for name, (_, times) in results.items()}
for name, c in cost.items():
    print(f"{name:28s} {1e6 * c:6.1f} microseconds per step")
for first, second in [
    ("rush_larsen", "rush_larsen_2"),
    ("generalized_rush_larsen", "generalized_rush_larsen_2"),
]:
    print(f"cost({second}) / cost({first}) = {cost[second] / cost[first]:.2f}")

# Finally, the wall time each scheme needs to reach a given accuracy,
# interpolated in log-log from the measurements above. We use the smallest
# error that all four schemes reach.

target = max(min(errors) for errors, _ in results.values())
print(f"time to reach relative error {target:.1e}")
for name, (errors, times) in results.items():
    # errors decrease with decreasing dt, so reverse to make them increasing
    t_target = np.exp(np.interp(np.log(target), np.log(errors[::-1]), np.log(times[::-1])))
    print(f"{name:28s} {t_target:.3f} s")

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

# A second-order step costs about twice as much as a first-order one, as the
# printed ratios show (close to 2 for both families), since the model is
# evaluated twice. In return, the second-order schemes reach a given accuracy
# faster: for the smallest error that all four schemes reach, about
# $2 \times 10^{-3}$, they need roughly half the wall time of the
# corresponding first-order scheme. This is a moderate accuracy; at the
# coarsest steps above, where the second-order schemes are no more accurate,
# there is no gain. The right panel of the figure shows the same trade-off
# over the whole range.
#
# ```{figure} second_order_rush_larsen.png
# Error against time step (left) and against wall time (right).
# ```
#
# ## Why not goss's GRL2?
#
# Earlier implementations (e.g. goss) used $f(y^*)$ instead of the midpoint
# linearization $f(y^*) + a^*(y_n - y^*)$ in the full step, and evaluated the
# second stage at $t_n$ instead of $t_n + \Delta t/2$. That variant is only
# first order on stiff, time-dependent problems; see the docstring of
# {py:func}`gotranx.schemes.generalized_rush_larsen_2`.
#
# ## References
#
# - J. Sundnes, R. Artebrant, O. Skavhaug, A. Tveito. A second-order algorithm
#   for solving dynamic cell membrane equations. *IEEE Trans. Biomed. Eng.*
#   56(10):2546–2548, 2009.
# - M. Perego, A. Veneziani. An efficient generalization of the Rush-Larsen
#   method for solving electro-physiology membrane equations. *ETNA*
#   35:234–256, 2009.
