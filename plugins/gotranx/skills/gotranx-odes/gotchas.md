# gotranx gotchas

Configurations that fail or silently give a worse result, and what to do instead. Composition-specific
gotchas are at the end of [composition.md](composition.md).

## A `pyproject.toml` somewhere above silently overrides command-line flags

**What happens:** `gotranx ode2py model.ode --scheme rush_larsen_2` generates other schemes (or other
options) than asked for.
**Why:** without `-c`, every `ode2*` command reads the nearest `pyproject.toml` that has a
`[tool.gotranx]` table, in the current directory or any parent, and values in that file take
precedence over the flags.
**Do instead:** generate in a folder with no such table above it (check with the loop in SKILL.md
Step 4), or pass `-c` with a config file you control, and put the options there.
<!-- verified: src/gotranx/cli/__init__.py (config_data.get(key, flag)), cli/utils.py find_pyproject_toml_config -->

## Calling `rhs` and a scheme with the same argument order

**What happens:** wrong results or shape errors, no exception that points at the cause.
**Why:** `rhs(t, states, parameters)` has time first; `<scheme>(states, t, dt, parameters)` has
states first. `monitor_values(t, states, parameters)` follows `rhs`.
**Do instead:** call by keyword where possible; when handing a scheme to a solver, check the order it
expects (fenicsx-beat calls `fun(states=..., t=..., parameters=..., dt=...)` by keyword).
<!-- verified: generated code from ode2py -->

## The CellML model's own stimulus fires in a tissue simulation

**What happens:** tissue is stimulated twice (by the PDE solver and by every cell), or a single-cell
run starts its first beat later than expected.
**Why:** CellML cell models carry a periodic stimulus current (`i_Stim_Amplitude`, `stim_amplitude`,
`IstimAmplitude`, … with a start time that differs between files: 0 ms in gotranx's ToR-ORd test
file, 50 ms in the O'Hara-Rudy 2011 file of the split-ode example).
**Do instead:** set the amplitude to 0 (`init_parameter_values(<name>=0.0)`) when the PDE stimulates;
for single cells, read the start/period parameters and set them for the protocol.
<!-- verified: cellml2ode of tests/cellml_files/ToRORd_dynCl_mid.cellml; split-ode example text -->

## numba fails to compile the generated functions

**What happens:** numba typing errors on the generated code.
**Why:** the default `--shape dynamic` checks at run time whether the input is 1D or 2D, which numba
cannot compile.
**Do instead:** generate with `--shape single` for numba (or `--shape multiple` for always-2D).
Keep singularity removal on: without it numba crashes on 0/0 at the removable poles, where numpy only
warns.
<!-- verified: examples/run-python/main_numba.py text -->

## Automatic gating detection exponentiates states you did not expect

**What happens:** `rush_larsen`/`rush_larsen_2` treat Markov-chain states (ToR-ORd IKr `C1`, `C2`,
`C3`, …) and some concentrations (ToR-ORd `cansr`) as gating states; their sum / conservation is no
longer exact.
**Why:** the detection is structural (states whose derivative is linear in themselves), not physical.
**Do instead:** print `gotranx.linearization.gating_states(ode)` and, if it matters for the study,
use `hybrid_rush_larsen_2` with an explicit `--stiff-states` list, or check the invariant over the
run at the chosen `dt`.
<!-- verified: gating_states on ToRORd_dynCl_mid -->

## `hybrid_rush_larsen` without stiff states is forward Euler

**What happens:** small stable `dt`, no benefit from the "Rush–Larsen" name.
**Why:** hybrid applies the exponential update only to the listed states; none listed means none.
**Do instead:** pass `--stiff-states` (repeat per state), or use `rush_larsen` (automatic detection).
Before gotranx 2.10, `get_scheme("rush_larsen")` returned this hybrid scheme — upgrade.
<!-- verified: CHANGELOG; v2.9.0 schemes.get_scheme maps rush_larsen to hybrid_rush_larsen -->

## `generalized_rush_larsen` is robust but not conservative

**What happens:** linear invariants (total ion content, Markov occupancies summing to 1) drift over
many beats.
**Why:** every state gets its own exponential update; the update is exact for linear decoupled
equations, but it breaks linear combinations that explicit schemes keep.
**Do instead:** for long pacing runs prefer `rush_larsen_2` (fenicsx-beat's default) at a `dt` where it
is stable; use GRL when stability at large `dt` matters more.
<!-- unverified: drift magnitude per model; from the fenicsx-beat 0.14 defaults study -->

## Regenerated code differs from code generated with an older gotranx

**What happens:** results change slightly after regenerating, especially near `v = 0` and for GHK
fluxes.
**Why:** removable singularities are guarded by default (since 2.9; the guards were extended to
the right-hand side and conditionals later); older code could have a wrong, even growing, linearization near the pole.
**Do instead:** keep the guards (default). Use `--no-remove-singularities` only to reproduce old
output exactly. Warnings such as "Dropping a guard whose replacement failed its numeric check" mean
that particular expression is emitted as written.
<!-- verified: CHANGELOG; warnings seen when composing ToR-ORd + Land. unverified: that those dropped guards are harmless -->

## A stale version number after `pip install -e`

**What happens:** `gotranx.__version__` reports an older version than the checkout.
**Why:** the version comes from installed metadata, written at install time.
**Do instead:** check features (`gotranx list-schemes` lists `rush_larsen_2`; `gotranx compose --help`
works), or re-run the editable install to refresh the metadata.
<!-- verified: editable install reporting 2.10.0 for a 2.11.0 checkout -->

## Including the generated C header in more than one file

**What happens:** linker error "multiple definition of `NUM_STATES`" (and of every function).
**Why:** `ode2c --to .h` defines non-`static` globals and functions in the header.
**Do instead:** include it in exactly one `.c` file and declare what others need, or use `ode2cpp`
(everything `inline`/`constexpr` in a namespace).
<!-- verified: gcc, two translation units including ode2c output -->

## UFL output fails on models with a periodic stimulus

**What happens:** `ode2ufl` raises a `KeyError` from inside sympy.
**Why:** UFL has no `floor`, `ceiling` or `Mod`, which periodic stimuli (most CellML models) use.
**Do instead:** remove the stimulus from the model (or replace it by a parameter set from outside)
before `ode2ufl`.
<!-- verified: examples/ufl_backend.py note -->

## Variables named exactly `Min` or `Max`

**What happens:** the model fails to parse (recent versions).
**Why:** `Min`/`Max` are now functions.
**Do instead:** rename them (`min`, `max`, `Max_x` are fine).
<!-- verified: CHANGELOG -->

## CytoZoo output does not load

**What happens:** the generated Julia module fails against a registered CytoZoo release.
**Why:** it calls `resolve_parameter` and `CytoZoo.primal`, which are not yet released.
**Do instead:** use a development checkout of CytoZoo.jl (as the CytoZoo example does), or
`ode2julia` for plain Julia.
<!-- verified: CHANGELOG, docs/cli.md -->

## `--no-cse` makes code slower

**What happens:** much larger generated scheme functions and slower steps.
**Why:** without common-subexpression elimination the linearized expressions are inlined.
**Do instead:** leave `--cse` on, except for the vectorized numpy backend over many cells, where each
temporary is a whole array and `--no-cse` can be faster — measure.
<!-- verified: docs/config.md -->

## `clang-format` warning when generating C/C++/CUDA

**What happens:** "Cannot apply clang-format, please install 'clang-format-docs'".
**Why:** formatting is optional; the code is written unformatted.
**Do instead:** ignore it, pass `--format none`, or install `gotranx[formatters]`.
<!-- verified: ode2c without clang-format -->

## The adaptive reference solver never fires the stimulus

**What happens:** a `solve_ivp` (or other adaptive) reference run of a cell model stays at rest, or
skips some beats.
**Why:** with no step-size limit the solver takes steps longer than the stimulus pulse (often 1–2 ms)
and never evaluates the right-hand side while it is on.
**Do instead:** pass `max_step` of at most half the stimulus duration (odezoo's model simulations use
≤ 0.25 ms and ≤ half the stimulus duration).
<!-- verified: odezoo tools/simulate_models.py _max_step -->
