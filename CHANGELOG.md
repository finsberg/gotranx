# Changelog

## 2.0.0

### Changed — this changes the numerics of generated code

`generalized_rush_larsen` and `hybrid_rush_larsen` now linearize on the true
diagonal of the Jacobian, `∂f_i/∂y_i`, differentiating through intermediate
expressions.

Previously the linearization was the derivative of the state derivative
*expression as written*. When a derivative was routed through an intermediate —
which is how essentially every cardiac model is written —
`sympy` correctly returned zero and the scheme silently emitted forward Euler
for that state. In practice this meant the exponential update reached the gating
variables only, and `generalized_rush_larsen` degenerated to the original 1978
Rush-Larsen method rather than the generalization of Sundnes et al. (2009) that
it cites.

`∂f_i/∂y_i` is a property of the right-hand side, not of how it is spelled, so
`I = g*(V - E); dV_dt = -I` and `dV_dt = -(g*(V - E))` now generate the same
scheme. They previously did not.

**If you regenerate an existing model, its numerics will change.** They will be
more accurate and considerably more stable — models with fast calcium buffering
could previously require a time step 20-40x smaller than the literature value —
but the output is not bit-identical to 1.8.0. There is no opt-out flag; pin
`gotranx==1.8.0` if you need to reproduce earlier output exactly.

Affected states in the bundled models, as a guide to the scale of the change:

| model | states linearized before | after |
|---|---|---|
| `ORdmm_Land` | 39/48 | 48/48 |
| `ToRORd_dyn_chloride` | 41/52 | 52/52 |
| `tentusscher_panfilov_2006_M_cell` | 13/19 | 19/19 |
| `beeler_reuter_1977` | 7/8 | 8/8 |
| `fitzhughnagumo` | 1/2 | 2/2 |
| `lorentz` | 3/3 | 3/3 (unchanged) |

### Fixed

- `hybrid_rush_larsen` applied its zero-derivative test after the stiff-state
  test and skipped the bookkeeping, so `--stiff-states V` produced forward Euler
  and then reported that `V` was "not found in the ODE". States that are present
  are now always reported as found, and an explicitly stiff state whose
  derivative is genuinely zero produces a warning instead.
- The diagnostic message read "where marked as stiff" and passed the state set
  as a `structlog` `extra=` key rather than formatting it into the message.

### Added

- `gotranx.linearization.diagonal_jacobian(ode, remove_unused=False)`, which
  returns `∂f_i/∂y_i` for every state by forward-mode automatic differentiation
  over the assignment graph.

### Notes for the generated code

Deep linearization makes the generated scheme measurably bigger and slower to
generate. Measured on `ToRORd_dyn_chloride`'s Python `generalized_rush_larsen`:

- Generated source: **1052 → 2754 lines**; zero-division guards **54 → 66**.
  A `d<state>_dt_linearized` local now appears for every state, alongside CSE
  temporaries named according to the `cse` strategy below
  (`_linearization_temp_<k>` under the default). Relevant if you post-process
  generated sources.
- Code generation time: **~3.6x slower** (0.79 s → 2.84 s under the default
  `cse="joint"`; 0.59 s of that is the AD sweep and most of the rest is CSE).
  Fine for a one-off codegen step, but CI benchmarks that time code
  generation itself will notice.
- **Peak live temporaries on the plain-numpy backend.** CSE introduces named
  locals for every factored-out subexpression, and CPython keeps every local
  bound until the function returns (unlike C, Julia, and JAX-under-`jit`,
  which reuse buffers once a compiler's own liveness analysis says a value is
  dead). Measured: distinct locals in the generated ToRORd function go
  **595 → 1011** (+70%) once deep linearization's CSE'd temporaries are
  added. For vectorized use over many cells (see
  `docs/vectorized_computations.md`) that is a real memory increase. The
  `cse` parameter below controls how many of those locals there are; `"none"`
  avoids the increase entirely, at the cost of far more operations.
- **CSE strategy is now selectable, and joint is the default.** Both
  `generalized_rush_larsen` and `hybrid_rush_larsen` take a `cse` parameter
  (`gotranx.schemes.CSEStrategy`: `"joint"`, `"per_state"`, `"none"`; plain
  strings work too), exposed on the CLI as `--cse` for `ode2py`, `ode2c`,
  `ode2julia` and `ode2ufl`, and as `cse` under `[tool.gotranx]` in
  `pyproject.toml`.

  - `"joint"` (**new default**) runs one `sympy.cse` across every state being
    linearized at once, so a subexpression shared *between* states is
    computed once rather than once per state. It replaces `"per_state"` as
    the default because it costs the same or fewer operations on every
    backend measured, for essentially the same number of live temporaries —
    the reasoning that made per-state CSE the previous default (that its
    temporaries die at the end of each state's own update) turned out not to
    pay off in practice: none of C, Julia, JAX-under-`jit`, or CPython itself
    actually frees a temporary early based on emission order, so per-state
    bought nothing over joint except a higher operation count.
  - `"per_state"` is the previous default, kept working. It is the better
    choice again if a future change starts emitting `del` for a temporary
    once it is dead, since a per-state temporary is provably dead at the end
    of that state's own update, where a joint one may still be needed by a
    later state.
  - `"none"` performs no CSE at all: every `d<state>_dt_linearized` is one
    fully inlined expression. It costs the most operations but produces the
    fewest named locals, which is what the peak-live-temporaries concern
    above actually tracks — this is the option that avoids that memory cost
    on the vectorized numpy backend.

  Measured operation count relative to the plain rhs, and number of
  temporaries, on `ToRORd_dyn_chloride` and on `base_model_IM.ode`:

  | strategy | ToRORd ops/rhs | ToRORd temporaries | base_model_IM ops/rhs | base_model_IM temporaries |
  |---|---|---|---|---|
  | `joint` (default) | 1.077x | 430 | 0.583x | 55 |
  | `per_state` | 1.352x | 405 | 0.686x | 49 |
  | `none` | 9.426x | 0 | 1.049x | 0 |

  Joint and per-state have essentially the same temporary count (see table
  above), so switching the default from one to the other does not change the
  595 → 1011 peak-live-temporaries measurement above.

  If you diff generated output across gotranx versions: regenerating an
  existing model now produces different temporary names than before (joint's
  shared pool is named `_linearization_temp_<k>`, not
  `_<derivative>_linearized_<k>`), and the different factoring reassociates
  floating-point operations, which can shift results at the ~1e-14 level.
