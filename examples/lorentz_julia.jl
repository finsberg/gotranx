# ---
# jupyter:
#   kernelspec:
#     display_name: Julia
#     language: julia
#     name: julia
# ---

# # Solving an ODE system in Julia
#
# In this example we will generate code for the Lorenz system and solve it in Julia.
#
# This page is a Julia script, `examples/lorentz_julia.jl`. To run it yourself you need the
# `gotranx` command on your `PATH`, and the packages in `examples/Project.toml`:
#
# ```bash
# julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# cd examples && julia --project=. lorentz_julia.jl
# ```
#
# First we write the ODE and save it to disk, in a file called `lorentz.ode`

ode = """
parameters(
sigma=12.0,
rho=21.0,
beta=2.4
)

states(x=1.0, y=2.0,z=3.05)

dx_dt = sigma * (y - x)
dy_dt = x * (rho - z) - y
dz_dt = x * y - beta * z
"""
write("lorentz.ode", ode);

# Next we generate the Julia code for the ODE system, with the generalized Rush-Larsen scheme

# + tags=["remove-output"]
run(`gotranx ode2julia lorentz.ode --scheme generalized_rush_larsen -o lorentz.jl`);
# -

# This writes a file called `lorentz.jl`. Including it defines the model's constants and
# functions: `NUM_STATES`, `init_state_values!`, `state_index`, `rhs`,
# `generalized_rush_larsen` and so on

include("lorentz.jl");

# We start from the initial values in the ODE file, and take steps of 0.01 up to $t = 100$.
# `generalized_rush_larsen` writes the next state into its last argument

# +
dt = 0.01
t = 0:dt:100

states = zeros(NUM_STATES)
init_state_values!(states)
parameters = zeros(NUM_PARAMS)
init_parameter_values!(parameters)

trajectory = zeros(length(t), NUM_STATES)
trajectory[1, :] = states
next_states = zeros(NUM_STATES)
for i in 2:length(t)
    generalized_rush_larsen(states, t[i - 1], dt, parameters, next_states)
    states .= next_states
    trajectory[i, :] = states
end
# -

# The state indices come from `state_index`, and are one-based

x, y, z = (trajectory[:, state_index(name)] for name in ("x", "y", "z"))
println("At t = $(t[end]): x = $(x[end]), y = $(y[end]), z = $(z[end])")

# Finally we plot the trajectory

using Plots
plot(x, y, z; xlabel = "x", ylabel = "y", zlabel = "z", legend = false)
