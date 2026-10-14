# Numerical schemes

`gotranx` can generate more than the right-hand side $f(t, y)$ of an ODE. It can also generate a function that takes **one time step** with a given numerical scheme. Such a function has the signature

```python
def scheme_name(states, t, dt, parameters): ...
```

and returns the state values at time $t + \Delta t$. This is useful when the ODE solver must be very cheap per step, for example when the ODE is solved at every node of a mesh in a coupled PDE–ODE simulation (see {doc}`../vectorized_computations`), or when the generated code is used from a language without a good ODE solver library.

Throughout this section we consider a system of $N$ ordinary differential equations

```{math}
\frac{\mathrm{d} y}{\mathrm{d} t} = f(t, y), \qquad y(t_0) = y_0, \qquad y = (y_1, \ldots, y_N),
```

discretized with a constant time step $\Delta t$ such that $t_n = t_0 + n \Delta t$ and $y^n \approx y(t_n)$.

## Available schemes

| Scheme | `--scheme` value | Order | Treats exponentially | Page |
|--------|------------------|-------|----------------------|------|
| Forward (explicit) Euler | `explicit_euler` | 1 | no state | {doc}`forward_euler` |
| Rush–Larsen | `hybrid_rush_larsen` with the gating variables as stiff states | 1 | the gating variables | {doc}`rush_larsen` |
| Generalized Rush–Larsen | `generalized_rush_larsen` | 1 | every state | {doc}`generalized_rush_larsen` |
| Hybrid Rush–Larsen | `hybrid_rush_larsen` | 1 | the states you choose | {doc}`hybrid_rush_larsen` |

You can list the schemes from the command line with

```shell
gotranx list-schemes
```

and generate code for one or more of them with the `--scheme` flag, e.g.

```shell
gotranx ode2py model.ode --scheme explicit_euler --scheme generalized_rush_larsen -o model.py
```

The same options can be set in a configuration file (see {doc}`../config`), and from Python with the `scheme` argument of the code generators, e.g.
{py:func}`gotranx.cli.gotran2py.get_code`.

## Which scheme should I use?

All four schemes are explicit, single-step and first-order accurate, so they cost roughly one evaluation of the right-hand side per step. They differ in **stability**:

- {doc}`forward_euler` is the simplest and cheapest scheme, but it is only stable when the time step is small compared to the fastest time scale of the model.
- The Rush–Larsen family integrates some or all states with an exponential update that is exact for linear equations. This removes the time step restriction coming from those states, at the cost of one exponential per state per step.
- {doc}`generalized_rush_larsen` applies the exponential update to every state and is the most robust choice. It is a good default for cardiac cell models.
- {doc}`hybrid_rush_larsen` applies it only to the states you list, so you only pay for the exponentials you need.

A larger stable time step does not mean a more accurate solution. All schemes are first order, and for a given $\Delta t$ forward Euler is sometimes more accurate than the exponential schemes when it is stable. Check the accuracy of your simulation by refining the time step.

## The running example

To compare the schemes we use the classical model of the squid giant axon by [Hodgkin and Huxley (1952)](https://doi.org/10.1113/jphysiol.1952.sp004764). It has a membrane potential $V$ and three gating variables $m$, $h$ and $n$. You can download the model in `.ode` format {download}`here <./hodgkin_huxley.ode>`.

```{literalinclude} hodgkin_huxley.ode
:language: text
```
