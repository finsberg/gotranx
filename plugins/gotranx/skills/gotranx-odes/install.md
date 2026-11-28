# Installing gotranx and what the backends need

gotranx is pure Python (Python ≥ 3.12) and installs with pip or conda:

```bash
python -m pip install gotranx            # or: conda install -c conda-forge gotranx
python -m pip install "gotranx[formatters]"   # adds clang-format for nicely formatted C/C++/CUDA
```

It brings sympy, myokit (CellML import/export), pint, lark, black and the CLI dependencies. No
compiler is needed to *generate* code. Recommend a virtual environment or conda environment the user
already uses for their simulations, so the downstream solver (fenicsx-beat, sknm, simcardemsx) sees
the same gotranx.

What each backend needs to *run* the generated code (check only what the plan uses):

| Target | Generate with | Needs at run time |
|---|---|---|
| Python, numpy | `gotranx ode2py` | numpy |
| Python, numba | `gotranx ode2py --shape single` | `python -m pip install numba` |
| Python, JAX | `gotranx ode2py --backend jax` | `python -m pip install jax` |
| C | `gotranx ode2c` | a C compiler (`gcc`/`clang`); `cmake` for the extension example |
| C++ | `gotranx ode2cpp` | a C++17 compiler |
| CUDA | `gotranx ode2cuda` | NVIDIA GPU and `nvcc` |
| Julia | `gotranx ode2julia` / `ode2mtk` | Julia (+ ModelingToolkit for `ode2mtk`) |
| CytoZoo | `gotranx ode2cytozoo` | Julia + a development checkout of CytoZoo.jl (see [gotchas.md](gotchas.md)) |
| UFL | `gotranx ode2ufl` | `fenics-ufl` (and dolfinx to assemble) |
| Markdown | `gotranx ode2md` | nothing |

Quick checks:

```bash
python -c "import numba; print(numba.__version__)"
gcc --version
which clang-format    # missing → code is written unformatted, with a warning; it still works
```

Models from the odezoo registry need `python -m pip install odezoo` (see [models.md](models.md)).

## Install gotchas

- **Stale version in an editable install.** `gotranx.__version__` comes from the installed metadata
  and can lag behind the checkout. Check features (`gotranx list-schemes` lists `rush_larsen_2`;
  `gotranx compose --help` works) rather than the number.
- **`clang-format` missing** only affects formatting: `ode2c`/`ode2cpp`/`ode2cuda` warn
  "Cannot apply clang-format" and write unformatted code. Pass `--format none` to silence it, or
  install the `formatters` extra.
- **Generated code does not import gotranx.** Generated Python needs only numpy (or numba/JAX), so a
  cluster job that only runs the generated file does not need gotranx installed — but the generated
  file must be regenerated when the model changes.
