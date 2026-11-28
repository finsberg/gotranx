---
name: gotranx-odes
description: Turn cardiac cell models and other ODE systems into code with gotranx — write or convert models (.ode, CellML, odezoo), compose models (e.g. electrophysiology + contraction) with a recipe and glue file, pick a numerical scheme (Rush-Larsen family, explicit Euler) and time step, and generate Python (numpy/numba/JAX), C, C++, CUDA, Julia/CytoZoo, UFL or Markdown. Use when the user wants to simulate a single cell, prepare a cell model for fenicsx-beat, sknm or simcardemsx, convert between CellML and .ode, couple two ODE models, choose a scheme or time step, or debug generated code that blows up or returns NaN.
---

# Generating code from ODE models with gotranx

`gotranx` parses ODE models written in its `.ode` language (or converted from CellML), builds a
symbolic (sympy) model, and generates code that evaluates the right-hand side and takes one time
step with a chosen scheme, in Python, C, C++, CUDA, Julia, UFL and more. Two ways to drive it:

- **CLI**: `gotranx cellml2ode`, `gotranx ode2py` / `ode2c` / …, `gotranx compose`,
  `gotranx list-schemes`. Good for one-off conversions and reproducible files on disk.
- **Python API**: `gotranx.load_ode`, `gotranx.compose`, `gotranx.cli.gotran2py.get_code`. Needed
  for splitting models, generating code inside a simulation script, or anything programmatic.

gotranx does not solve tissue problems: it produces the cell-model code that fenicsx-beat, sknm and
simcardemsx consume.

Your job: find out what the user really wants, agree on an approach with them, build it, and check
it before they rely on it. Work through the five steps below in order. Ask **one question at a
time** (use your agent's question tool if it has one, otherwise ask in plain text). Do not skip the
interview because the request looks clear — the scheme, backend and composition choices change the
result.

Reference files (read when a step needs them):

- [install.md](install.md) — installing gotranx and what the backends need
- [models.md](models.md) — where models come from: `.ode` files, CellML, odezoo
- [composition.md](composition.md) — coupling two or more models (e.g. EP + contraction)
- [schemes.md](schemes.md) — choosing a scheme and a time step
- [targets.md](targets.md) — each backend: command, generated API, how to call it
- [gotchas.md](gotchas.md) — configurations that fail, and settings that work better

## Step 1: Check the environment

```bash
python -c "import gotranx; print(gotranx.__version__)"
gotranx list-schemes
```

- **Version.** This skill describes gotranx 2.11 or later (`gotranx compose` is new in 2.11,
  the second-order schemes in 2.10). An editable install can report a stale `__version__`, so check
  features: `gotranx list-schemes` must list `rush_larsen_2`, and `gotranx compose --help` must
  work. If both work, the installed gotranx is new enough whatever the number says. If they do not,
  recommend upgrading (`python -m pip install -U gotranx`) or follow that version's documentation
  instead of this skill.
- Later, once the plan is clear, check what the chosen backend needs ([install.md](install.md)):
  a C compiler for C, `nvcc` for CUDA, numba or JAX for fast Python, Julia for Julia/CytoZoo,
  `odezoo` for registry models.
- **Never install anything without asking.** Show the command and let the user run or approve it.

## Step 2: Interview the user

Ask one question at a time, multiple choice with an "other" option, and say why it matters when that
is not obvious. Skip a topic the user already answered (confirm instead), and skip topics that do not
apply. For a narrow debugging request ("my generated code gives NaN", "which scheme?") ask only what
changes the answer — target `dt`, single cell or tissue, which scheme they used — then reproduce the
problem before proposing fixes. Topics, in order:

1. **Goal** — single-cell simulation (action potential, pacing to steady state, drug block);
   a cell model for a tissue solver (fenicsx-beat, simcardemsx, sknm); many cells on a GPU; a
   Julia/CytoZoo workflow; documenting a model (Markdown/LaTeX); converting a model between CellML
   and `.ode`; coupling two models (electrophysiology + contraction, or other parts). The goal
   decides the backend and whether a scheme is needed at all.
2. **Model source** — their own `.ode` file; a CellML file or Physiome Model Repository link; a model
   from the odezoo registry; several models to couple. See [models.md](models.md). If they bring
   **two or more models to couple**, go through the composition questions in
   [composition.md](composition.md) now (coupling direction, variables that cross, units of each
   part, one model or split halves, who provides stretch).
3. **Target** — Python (plain numpy, numba, JAX), C (single-cell program or a Python extension),
   C++, CUDA, Julia / ModelingToolkit / CytoZoo, UFL (inside dolfinx forms), Markdown. If the
   consumer is fenicsx-beat, sknm or simcardemsx, the target is Python (numpy) unless that package's
   docs say otherwise. See [targets.md](targets.md).
4. **Time stepping** — a generated one-step scheme with a fixed `dt`, or only the right-hand side for
   an adaptive solver (scipy `solve_ivp` with BDF/LSODA, Julia's solvers). For a scheme: how
   accurate (first or second order — second order pays off only with Strang splitting downstream),
   which `dt` they plan, and how long they pace (a few beats, or many beats to a steady state: errors
   that grow every beat matter only then). See [schemes.md](schemes.md).
5. **Downstream details** — where the stimulus is applied (the model's own stimulus, or the tissue
   solver's), which parameters they will vary (pass them via `init_parameter_values`), which
   intermediate values they need to read (currents, active tension: monitored values), and whether
   the model must run vectorized over many cells.
6. **Validation** — do they have a reference trace (paper figure, another implementation) to compare
   against?

## Step 3: Propose alternatives

Present **2–3 approaches with trade-offs and a recommendation**. Typical axes:

- **Scheme**: `rush_larsen_2` (second order, gating states exponential; fenicsx-beat's default) vs
  `generalized_rush_larsen` (every state exponential, most robust at large `dt`, less accurate at
  the same `dt`) vs `explicit_euler` (cheapest per step, needs a small `dt`) vs right-hand side + an
  adaptive stiff solver (reference quality, slow, not for tissue). **A model with a Markov chain
  (ToR-ORd's IKr) paced for many beats** drifts with both Rush–Larsen defaults at 0.05 ms (APD90
  +65 ms with `rush_larsen_2`, −21 ms with GRL after 200 beats); offer `hybrid_rush_larsen_2`
  without the chain at `dt` ≤ 0.025 ms, or `rush_larsen_2` at `dt` ≤ 0.01 ms. Details:
  [schemes.md](schemes.md#markov-chains-and-automatic-gating-detection).
- **Backend**: plain numpy (simple, vectorized over cells) vs numba (fast loops, needs
  `--shape single`) vs a C extension (fastest on CPU, needs a compiler) vs CUDA (many cells on a
  GPU). Details: [targets.md](targets.md).
- **Composition** (when there are several models): one composed model vs composed-then-split halves
  vs separate models coupled by hand. Details: [composition.md](composition.md).
- **Interface**: CLI (files on disk, reproducible, easy to diff) vs Python API (needed for split
  models, generating code inside a simulation script, or parameter sweeps).

Then **summarize the chosen setup** in a few lines (model source, composition, scheme, `dt`,
backend, output file names, stimulus handling) and get an explicit yes before building.

## Step 4: Build

Work in a fresh folder the user agrees on (suggest one, e.g. `./gotranx-<model>/`), **without a `pyproject.toml` that has a `[tool.gotranx]`
table in it or in any parent folder** — such a table silently overrides command-line flags
([gotchas.md](gotchas.md)). Check with

```bash
d=$PWD; while [ "$d" != / ]; do grep -l "tool.gotranx" "$d/pyproject.toml" 2>/dev/null; d=$(dirname "$d"); done
```

(no output means none), or pass `-c <file>` with a config file you control.

**CLI route**, e.g.:

```bash
gotranx cellml2ode model.cellml -o model.ode
gotranx ode2py model.ode --scheme rush_larsen_2 -o model.py
```

**Python route**, e.g.:

```python
import gotranx

ode = gotranx.load_ode("model.ode")
code = gotranx.cli.gotran2py.get_code(ode, scheme=[gotranx.schemes.Scheme.rush_larsen_2])
model = {}
exec(code, model)  # or write `code` to model.py and import it
```

Before using a gotranx function, read its signature in the installed version
(`python -c "import gotranx, inspect; print(inspect.signature(gotranx.cli.gotran2py.get_code))"`,
`gotranx ode2py --help`) — options change between versions.

Generating code for a large model takes a while: ToR-ORd (45 states) takes ~15 s for `ode2py`, most
of it in sympy; `--format none` skips the formatter. Warnings "Dropping a guard whose replacement
failed its numeric check" for `nai`, `ki`, `cli` or `clss` come from Nernst potentials
(`log(c_o/c_i)` at `c_i = 0`, a logarithmic singularity that cannot be guarded) and are harmless —
tell the user what they mean, do not hide them ([gotchas.md](gotchas.md)).

Find the names you will need in the generated module, not by guessing:

```python
import model

print([k for k in model.parameter if "stim" in k.lower()])  # the model's own stimulus parameters
print(list(model.state))  # find the membrane potential (v, V, V_m, membrane_V, ...)
```

If no parameter has "stim" in its name, look for the stimulus component in the `.ode` file
(`grep -n -i -A6 "stim" model.ode`): ORdmm_Land, for instance, has `amp` and `duration` in a
component `Istim`, and a single pulse at the start (`Conditional(Le(t, duration), amp, 0)`) instead
of a periodic one.

When adapting a gotranx example:

- it reads models from the repository or from odezoo — download them or point to the user's file,
  and use absolute paths;
- its end time and plotting are cut for the docs build — set them for the user's goal;
- parameter values it sets (e.g. Land's intact-cell parameters in the split-ode example) are part of
  the physics — keep them or replace them knowingly. Set them in a glue file or at run time, never by
  editing a generated or composed `.ode` by hand: regenerating it loses the edit
  ([composition.md](composition.md)).

## Step 5: Check, then hand over

1. **Smoke check** the generated code (seconds): import it, build `init_state_values()` and
   `init_parameter_values()`, run one beat (or one period of the user's protocol) at the chosen
   `dt`, and check: all values finite; the membrane potential makes an upstroke (from rest to above
   about 0 mV for physiological models) when stimulated and returns near rest; states that must stay
   in [0, 1] (gates, occupancies) do. Then **compare** against a reference: the same run at `dt/4`,
   or `rhs` with `scipy.integrate.solve_ivp(method="BDF", rtol=1e-8)` and `max_step` at most half
   the stimulus duration (otherwise it can step over the stimulus, [schemes.md](schemes.md)). Report the difference in
   action-potential duration or peak, not only "it runs". As a rule of thumb, an APD90 within ~1 %
   and a peak within ~1 mV of the reference is fine for tissue work; ask the user what they need.
   Measure from the stimulus onset (the model's own stimulus may start at 0 ms or 50 ms) and keep the
   stimulus edges on the time grid (`dt` dividing the pulse duration), otherwise the error is
   dominated by when the stimulus switches and the expected convergence order does not show.
   APD90: the time from the upstroke until `v` has come back 90 % of the way from its peak to the
   resting value (the value at the end of the beat). If the run at the user's `dt` gives NaN (typical
   for `explicit_euler` through the upstroke), compare the candidate schemes at that `dt` against the
   adaptive reference instead of against `dt/4`.
2. **Multi-beat check** when the model will be paced for more than a few beats (tissue, steady
   state, drug studies): run 50–100 beats at the chosen `dt` and with a reference (`dt/10`, or a
   scheme that keeps the model's invariants), compare APD90 at the last beat, and track linear
   invariants such as a Markov chain's occupancies summing to 1. A first beat that matches does not
   rule out an error that grows every beat: ToR-ORd with `rush_larsen_2` at 0.05 ms is within 1 ms
   on beat 1 and 65 ms off after 200 beats ([schemes.md](schemes.md#markov-chains-and-automatic-gating-detection)).
   A ToR-ORd beat takes a few seconds in generated numpy code, so ask before starting this run.
3. For composed models, also check the coupled outputs (e.g. active tension rises after the calcium
   transient and returns to baseline) and print what each part needs from the others
   (`gotranx compose` prints it; in Python, `missing_variables` of each half).
4. **Hand over**: the exact commands or script, the generated files and their API (which function to
   call with which argument order, [targets.md](targets.md)), the `dt` it was checked at, and how
   to regenerate after editing the model.

**Never start a long run without asking** — offer the command; run it only on an explicit yes.

## Related skills

- **odezoo-models** — finding and fetching published cell and contraction models; otherwise see
  https://computationalphysiology.github.io/odezoo.
- **beat-simulation** — using the generated model in a fenicsx-beat tissue simulation; otherwise
  see https://finsberg.github.io/fenicsx-beat.
- **sknm-simulation** — cell-network (Kirchhoff network) simulations with a generated membrane
  model; otherwise see https://github.com/ComputationalPhysiology/sknm.
- **simcardemsx-simulation** — electromechanics with a composed EP + contraction model; otherwise
  see https://computationalphysiology.github.io/simcardemsx.
- **crossbridge-models** — contraction models to compose with an EP model; otherwise see
  https://computationalphysiology.github.io/crossbridge.

Use a related skill if it is available; if not, follow the link.

## Ground rules

- **Do not invent API, CLI options or scheme names.** Check `--help`, `gotranx list-schemes`,
  `inspect.signature`, or the docs (https://finsberg.github.io/gotranx). If something does not
  exist, say so and offer the closest thing that does.
- Argument orders differ: `rhs(t, states, parameters)` but a scheme is
  `scheme(states, t, dt, parameters)`. Pass by keyword where the consumer allows it.
- Units: gotranx keeps the model's units. Check the time unit of every model (ms vs s) and the
  concentration units (mM vs µM) before coupling or choosing `dt`.
- Top pitfalls (all in [gotchas.md](gotchas.md)): a `[tool.gotranx]` table in a parent
  `pyproject.toml` overriding flags; a CellML model's own stimulus firing in a tissue simulation;
  numba needing `--shape single`; automatic gating detection breaking Markov chains, which makes the
  action potential drift over many beats; units and double-counted troponin when composing.
