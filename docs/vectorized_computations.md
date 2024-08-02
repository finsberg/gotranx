# Vectorized computations

One particular important application within cardiac electrophysiology is coupled ODE-PDE systems. For example the heart has an electrical current that flows through the heart which initiates contraction. This current can be modelled using reaction diffusion equations, where the reaction term is governed by an ODE model. In this case we would have one ODE for each degree of freedom in our geometry in which case we want to solve thousands or even million ODEs at the same time.

There is a library called [`fenics-beat`](https://finsberg.github.io/fenics-beat) that implements this functionality and that relies heavily on `gotranx` for code generation.
