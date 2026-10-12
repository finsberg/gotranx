# ---
# jupyter:
#   kernelspec:
#     display_name: Julia
#     language: julia
#     name: julia
# ---

# # Running a model in CytoZoo.jl
#
# [CytoZoo.jl](https://github.com/DerangedIons/CytoZoo.jl) is a Julia package of cardiac cell
# models that share one interface: a model is a callable struct with the
# DifferentialEquations.jl signature `f(du, u, p, t)`, which knows its states, parameters and
# monitored values by name. `gotranx ode2cytozoo` generates such a model from an ODE file, so
# the model can be solved with any DifferentialEquations.jl solver, and coupled to other
# CytoZoo models.
#
# In this example we generate the O'Hara-Rudy model with the Land model of contraction
# (`ORdmm_Land.ode`), solve one beat, and see what blocking half of the L-type calcium current
# does to the action potential and the active tension.
#
# This page is a Julia script, `examples/cytozoo/main.jl`. To run it yourself you need the
# `gotranx` command on your `PATH`, and the packages in `examples/Project.toml`:
#
# ```bash
# julia --project=examples -e 'using Pkg; Pkg.instantiate()'
# cd examples/cytozoo && julia --project=.. main.jl
# ```
#
# First we generate the model. `--model-name` names its struct

# + tags=["remove-output"]
run(`gotranx ode2cytozoo ORdmm_Land.ode -o ordmm_land.jl --model-name ORdmmLand`);
# -

# The generated file defines names such as `rhs` and `NUM_STATES` at its top level, so we
# include it in a module of its own

# +
using CytoZoo

module Generated
include("ordmm_land.jl")
end;
# -

# Constructing the struct gives the model with the parameters and initial values of the ODE
# file

model = Generated.ORdmmLand()
num_states(model), num_parameters(model)

# The model is the right-hand side, and CytoZoo builds an `ODEProblem` from it. The ODE file
# applies a stimulus during the first half millisecond, so a second of simulation is one beat.
# Cardiac cell models are stiff, so we use an implicit solver

using OrdinaryDiffEqBDF
problem = ODEProblem(model, (0.0, 1000.0))
control = solve(problem, FBDF(); abstol = 1e-8, reltol = 1e-8)
control.retcode

# Parameters live in a vector on the struct, and `parameter_index` finds one by name. Here we
# halve the L-type calcium current

blocked = Generated.ORdmmLand()
blocked.parameters[parameter_index(blocked, :scale_drug_ICaL)] = 0.5
drug = solve(ODEProblem(blocked, (0.0, 1000.0)), FBDF(); abstol = 1e-8, reltol = 1e-8)
drug.retcode

# The membrane potential is a state, and the active tension `Ta` is one of the monitored
# values, which `monitor_history` recomputes along a solution

# +
using Plots

v = transmembrane_potential_index(model)

function tension(solution, cell)
    history = monitor_history(solution, cell)
    return history.t, history.values[findfirst(==(:Ta), history.names), :]
end

potential = plot(xlabel = "Time (ms)", ylabel = "v (mV)")
active = plot(xlabel = "Time (ms)", ylabel = "Ta (kPa)")
for (solution, cell, label) in ((control, model, "control"), (drug, blocked, "ICaL × 0.5"))
    plot!(potential, solution.t, [u[v] for u in solution.u]; label)
    plot!(active, tension(solution, cell)...; label)
end
plot(potential, active; layout = (2, 1), size = (700, 600))
# -

# With less calcium entering the cell, the action potential is shorter, and the peak active
# tension is about 40 % lower.
