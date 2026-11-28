# Backends: command, generated API, how to call it

The code generators `ode2py`, `ode2c`, `ode2cpp`, `ode2cuda`, `ode2julia` and `ode2ufl` accept
`--scheme` (repeatable), `--remove-unused`, `--no-remove-singularities`, `-o/--outname` and
`-c/--config`. `ode2mtk` and `ode2cytozoo` generate no scheme (no `--scheme`), and `ode2md` has
neither `--remove-unused` nor `-c/--config`. Check the rest with `gotranx <command> --help`. The CLI
guide: https://finsberg.github.io/gotranx/docs/cli.html.

## Python (numpy) — `gotranx ode2py`

```bash
gotranx ode2py model.ode --scheme rush_larsen_2 -o model.py
```

Python API: `gotranx.cli.gotran2py.get_code(ode, scheme=[...], stiff_states=..., delta=..., cse=...,
format=..., missing_values=..., shape=..., backend=...)` returns the source as a string (write it to a
file, or `exec` it into a dict). Singularity removal is chosen when loading, not here:
`gotranx.load_ode(path, remove_singularities=False)`.

The generated module contains:

| Name | What |
|---|---|
| `parameter`, `state`, `monitor` | dicts name → index |
| `parameter_index(name)`, `state_index(name)`, `monitor_index(name)` | index lookup (raises `KeyError` for unknown names) |
| `init_parameter_values(**values)`, `init_state_values(**values)` | default arrays, with overrides by name (unknown names raise `KeyError`) |
| `rhs(t, states, parameters)` | right-hand side — note: **`t` first** |
| `monitor_values(t, states, parameters)` | all intermediates (currents, active tension, …), index with `monitor_index` |
| `<scheme>(states, t, dt, parameters)` | one step — note: **`states` first** |
| `missing_values(...)` | only for split models: what the other half needs |

Split models (see [composition.md](composition.md)) take the missing variables as an extra last
argument: `rhs(t, states, parameters, missing_variables)`,
`<scheme>(states, t, dt, parameters, missing_variables)`.

**Vectorized:** with the default `--shape dynamic`, `states` can be 1D `(num_states,)` or 2D
`(num_states, n_cells)` (state-major: one column per cell); parameters can be 1D (shared) or 2D.

```python
import numpy as np
import model

# names are the model's own: list them with [k for k in model.parameter if "stim" in k.lower()]
p = model.init_parameter_values(i_Stim_Amplitude=0.0)  # ToR-ORd's stimulus amplitude
y = model.init_state_values()
t, dt = 0.0, 0.05
for _ in range(int(1000 / dt)):
    y = model.rush_larsen_2(y, t, dt, p)
    t += dt
v = y[model.state_index("v")]
```

**numba:** generate with `--shape single` (the dynamic shape check is not numba-compatible), then
`numba.njit(model.rush_larsen_2)`. Example:
https://finsberg.github.io/gotranx/examples/run-python/main_numba.html (source:
[examples/run-python/main_numba.py](https://github.com/finsberg/gotranx/blob/main/examples/run-python/main_numba.py)).

**JAX:** `gotranx ode2py model.ode --backend jax ...`.

### Consumers

- **fenicsx-beat**: `exec` the code from `get_code` into a dict (or import the generated file) and
  pass the scheme function as `fun=` to beat's ODE solver (`beat.odesolver.DolfinODESolver`, with
  `init_states=`, `parameters=`, `num_states=`, and `v_index=` the index of the membrane potential,
  `model.state_index("v")`); beat calls it by keyword: `fun(states=..., t=..., parameters=..., dt=...)`.
  Set the model's own stimulus amplitude to 0 when beat's PDE stimulates. beat also has a C backend
  that runs gotranx itself. Check the signature in the installed beat; for the rest use the
  beat-simulation skill, or https://finsberg.github.io/fenicsx-beat.
- **sknm**: wraps a generated numpy module (`sknm.membrane.from_gotranx(module, v_name=...)`); use
  the sknm-simulation skill, or https://github.com/ComputationalPhysiology/sknm.
- **simcardemsx**: composed EP + contraction models; use the simcardemsx-simulation skill, or
  https://computationalphysiology.github.io/simcardemsx.

## C — `gotranx ode2c`

```bash
gotranx ode2c model.ode --scheme rush_larsen_2 --to .h -o model.h
```

Functions: `init_parameter_values(double *parameters)`, `init_state_values(double *states)`,
`rhs(t, states, parameters, values)`, `monitor_values(t, states, parameters, values)`,
`<scheme>(states, t, dt, parameters, values)` (writes the new states into `values`), index lookups,
and globals `NUM_STATES`, `NUM_PARAMS`, `NUM_MONITORED`.

The header defines non-`static` globals and functions: **include it in exactly one translation
unit**, or linking fails with "multiple definition of `NUM_STATES`". For several translation units
use C++ (`ode2cpp`, everything `inline` in a namespace). Examples: single-cell program
[examples/c-single-cell](https://github.com/finsberg/gotranx/tree/main/examples/c-single-cell),
JIT-compiled Python extension
[examples/compile-c-extension](https://github.com/finsberg/gotranx/tree/main/examples/compile-c-extension).

## C++ — `gotranx ode2cpp`

```bash
gotranx ode2cpp model.ode --scheme rush_larsen_2 --namespace mymodel -o model.hpp
```

A C++17 header, everything `inline`/`constexpr` in a namespace (default: the ODE's name);
`--float-type float` for single precision. Example:
[examples/cpp-single-cell](https://github.com/finsberg/gotranx/tree/main/examples/cpp-single-cell).

## CUDA — `gotranx ode2cuda`

```bash
gotranx ode2cuda model.ode --scheme generalized_rush_larsen -o model.cuh
```

One kernel per function, one thread per cell, `n_cells` as the last argument, state-major layout
(`states[i * n_cells + cell]`), parameters shared by all cells. Only includable from `.cu` files
compiled with `nvcc`. With single precision (`--float-type float`) compare a few cells against
double precision (or the Python backend) before trusting it. Example:
[examples/cuda-cable](https://github.com/finsberg/gotranx/tree/main/examples/cuda-cable).

## Julia, ModelingToolkit, CytoZoo — `gotranx ode2julia`, `ode2mtk`, `ode2cytozoo`

```bash
gotranx ode2julia model.ode --scheme generalized_rush_larsen -o model.jl
gotranx ode2cytozoo model.ode --v-name V -o model_cz.jl
```

`ode2cytozoo` output needs a CytoZoo.jl version that provides `resolve_parameter` and
`CytoZoo.primal`, **not yet released** — use a development checkout (see the CLI guide).

## UFL — `gotranx ode2ufl`

UFL expressions for writing the ODE into a FEniCSx variational form (and differentiating it with
`ufl.diff`). UFL has no `floor`, `ceiling` or `Mod`: a model using them — including the periodic
stimulus of most CellML models — fails (currently with a `KeyError` from sympy). Remove the
periodic stimulus from the model first. Example:
https://finsberg.github.io/gotranx/examples/ufl_backend.html (source:
[examples/ufl_backend.py](https://github.com/finsberg/gotranx/blob/main/examples/ufl_backend.py)).

## Markdown — `gotranx ode2md`

```bash
gotranx ode2md model.ode -o model.md
```

Equations and parameter tables for documentation.
