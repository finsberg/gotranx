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

## Automatic gating detection breaks Markov chains, and the error grows every beat

**What happens:** with `rush_larsen`/`rush_larsen_2`, a model with a Markov chain (ToR-ORd's IKr:
`C1`, `C2`, `C3`, `O_`, `I_`) loses the chain's sum-to-one over many beats, and the action potential
changes with it. ToR-ORd with `rush_larsen_2` at 0.05 ms, 200 beats at 1 Hz: sum 0.78, APD90
350 → 416 ms (converged: 351 ms). The first beat is fine, so a one-beat check passes.
**Why:** the detection is structural (states whose derivative is linear in themselves), not
physical. Each chain state gets its own exponential update, which does not keep their sum.
**Do instead:** print `gotranx.linearization.gating_states(ode)`. If a chain is in it, use
`hybrid_rush_larsen_2` with the detected states minus the chain (for ToR-ORd stable at
`dt` ≤ 0.025 ms, NaN at 0.04 ms), or `rush_larsen_2` at `dt` ≤ 0.01 ms (APD90 +3 ms after 200
beats). Check APD90 and the sum over 50+ beats ([schemes.md](schemes.md)).
<!-- verified: ToRORd_dynCl_mid, 200 beats of generated C at several dt, for rush_larsen_2, generalized_rush_larsen and hybrid_rush_larsen_2 -->

## `hybrid_rush_larsen_2` NaN after leaving states out

**What happens:** NaN within the first beat after removing states (e.g. a Markov chain) from the
`--stiff-states` list.
**Why:** every state not listed is stepped with forward Euler, and a stiff one is then unstable at
the `dt` that worked before. ToR-ORd's IKr chain needs `dt` ≤ 0.025 ms this way.
**Do instead:** lower `dt` until one beat is finite and close to the reference, or keep the state in
the list.
<!-- verified: ToRORd_dynCl_mid, hybrid_rush_larsen_2 without the chain: NaN at 0.04 and 0.05 ms, stable at 0.025 ms -->

## `hybrid_rush_larsen` without stiff states is forward Euler

**What happens:** small stable `dt`, no benefit from the "Rush–Larsen" name.
**Why:** hybrid applies the exponential update only to the listed states; none listed means none.
**Do instead:** pass `--stiff-states` (repeat per state), or use `rush_larsen` (automatic detection).
Before gotranx 2.10, `get_scheme("rush_larsen")` returned this hybrid scheme — upgrade.
<!-- verified: CHANGELOG; v2.9.0 schemes.get_scheme maps rush_larsen to hybrid_rush_larsen -->

## `generalized_rush_larsen` is robust, but less accurate and not conservative either

**What happens:** larger errors than `rush_larsen_2` at the same `dt` (action-potential peak
+4.6 mV for TP06 + Land and +4 mV for ToR-ORd at 0.05 ms, +8 mV for ORdmm_Land at 0.1 ms), and
linear invariants drift over many beats (ToR-ORd's IKr Markov sum 1.10 after 200 beats at 0.05 ms,
APD90 351 → 330 ms). Ion concentrations stay within about 1 % of the reference after 100 beats.
**Why:** every state gets its own exponential update: exact for a linear, decoupled equation, but
it breaks linear combinations of states.
**Do instead:** use GRL when stability at a large `dt` matters most, and compare against a smaller
`dt`. Switching to `rush_larsen_2` does not fix the drift for a model with a Markov chain: it is
worse there (entry above).
<!-- verified: TP06 + Land and ORdmm_Land, one beat against a fine reference; ToRORd_dynCl_mid, 100 and 200 beats of generated C -->

## Regenerated code differs from code generated with an older gotranx

**What happens:** results change slightly after regenerating, especially near `v = 0` and for GHK
fluxes.
**Why:** removable singularities are guarded by default since 2.1, in the right-hand side and in
the Rush–Larsen linearization (and inside `Conditional` branches since 2.6). Code generated before
could have a wrong, even growing, linearization near the pole.
**Do instead:** keep the guards (default). Use `--no-remove-singularities` only to reproduce old
output exactly.
<!-- verified: git tags (d9f215c "guard removable singularities ... by default" first in v2.1.0; 8ba543c "guard a pole inside a conditional" first in v2.6.0) -->

## "Dropping a guard whose replacement failed its numeric check"

**What happens:** this warning when loading or generating ToR-ORd (`var=nai`, `ki`, `cli`, `clss`),
ORdmm_Land (`nai`, `ki`) or a model composed from them.
**Why:** the candidates are Nernst potentials, `R*T/(z*F)*log(c_o/c_i)` at `c_i = 0`: a logarithmic
singularity, not a removable pole, so no guard exists and the numeric check rightly rejects one. The
expression is kept as written, and concentrations never reach 0. It is harmless.
**Do instead:** tell the user what it means; do not hide it. For a variable that is not a
concentration inside a `log`, read the expression (`grep -n "<var>" model.ode`): it stays unguarded
at that value, so check that the model cannot reach it.
<!-- verified: printed the rejected expressions from singularities._pole_at for ToRORd_dynCl_mid; same warnings for ORdmm_Land -->

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
