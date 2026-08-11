# # Solving an ODE system in Julia
#
# In this example we will generate code for the Lorenz system and solve it in Julia.
#
# First we write the ODE and save it to disk

ode_str = """
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

# We can now save the file to disk in a file called `lorentz.ode`

# +
from pathlib import Path

Path("lorentz.ode").write_text(ode_str)
# -

# Next we will generate the code for the ODE system

# + language="bash"
# gotranx ode2julia lorentz.ode --scheme generalized_rush_larsen -o lorentz.jl
# -

# This will generate a file called `lorentz.jl`. We can now solve the ODE system in Julia using the following code

# +
# include("lorentz.jl")
#
# dt = 0.01
# states = zeros(NUM_STATES)
# values= zeros(NUM_STATES)
# init_state_values!(states)
# p = zeros(NUM_PARAMS)
# init_parameter_values!(p)
# t = 0:dt:100
# x_index = state_index("x")
# y_index = state_index("y")
# z_index = state_index("z")
#
#
# x = [states[x_index]]
# y = [states[y_index]]
# z = [states[z_index]]
#
# for i in 1:length(t)
#     generalized_rush_larsen!(states, t[i], dt, p, values)
#
#     for i in 1:NUM_STATES
#         states[i]  = values[i]
#     end
#     push!(x, states[x_index])
#     push!(y, states[y_index])
#     push!(z, states[z_index])
# end
#
# println("Finished simulation")
# println("Last x, y, z: ", states[x_index], states[y_index], states[z_index])
#
# # Assuming you have "Plots" installed, you can plot the results
# using Plots
# plot(x, y, z)
# savefig("lorentz-julia.png")
# -
