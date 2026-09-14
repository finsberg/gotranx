# Changelog

## Unreleased

### Changed — this changes generated code

Removable singularities are now guarded by default, in the right-hand side as
well as in the Rush-Larsen linearization. Every assignment with a removable
pole in a state variable -- a gate rate such as
`(V + 10)/(exp((V + 10)/10) - 1)`, or a GHK flux divided by `exp(vfrt) - 1` --
is rewritten in that state and wrapped in a `Piecewise` whose other branch is
a truncated Taylor series of order 3 over a small window around the pole.

This matters most for the linearization. Since 2.0.0 the Rush-Larsen schemes
differentiate through intermediates, which turns a removable pole into a
double pole: ToRORd's `d(dv_dt)/dv` was +30.1 at `v = -1e-9` and raised at
`v = 0`, against a true value near -0.0405, and every action potential
crosses `v = 0` twice. A positive linearization turns the exponential update
from decaying to growing.

**If you regenerate a model with such a pole, its output changes** -- including
`rhs`, for ToRORd_dyn_chloride and ORdmm_Land, whose GHK poles were not guarded
before. Pass `--no-remove-singularities` (or `remove_singularities=False` to
`load_ode`, or `remove_singularities = false` in the config file) to emit the
expressions exactly as written. `ode2cellml` never guards: it exports a model,
not code.

### Fixed

- The previous mechanism (`ODE.remove_singularities`, reachable only from
  Python) discarded every GHK pole as infinite, because it looked for poles in
  whatever intermediate the model author had named; replaced a pole by a
  constant, which differentiates to zero and left the linearization wrong;
  used `sympy.limit`, which returns 0 for float-coefficient gate rates such as
  `0.2*(V + 23)/(1 - exp(-0.04*(V + 23)))` where the true limit is 5; and
  tested for the pole with exact equality. It has been removed.
- `sympy.cse` in the Rush-Larsen schemes no longer hoists subexpressions out of
  `Piecewise` branches, where a hoisted temporary could be `nan` at exactly
  the value the branch exists to avoid.

### Added

- `gotranx.singularities`, and `--remove-singularities/--no-remove-singularities`
  on `ode2py`, `ode2c`, `ode2julia`, `ode2mtk`, `ode2ufl` and `ode2md`.

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
  A `d<state>_dt_linearized` local now appears for every state, alongside
  `_linearization_temp_<k>` CSE temporaries. Relevant if you post-process
  generated sources.
- Code generation time: **~3.6x slower** (0.79 s → 2.84 s; 0.59 s of that is
  the AD sweep and most of the rest is CSE). Fine for a one-off codegen step,
  but CI benchmarks that time code generation itself will notice.
- **Peak live temporaries on the plain-numpy backend.** CSE introduces named
  locals for every factored-out subexpression, and CPython keeps every local
  bound until the function returns (unlike C, Julia, and JAX-under-`jit`,
  which reuse buffers once a compiler's own liveness analysis says a value is
  dead). Measured: distinct locals in the generated ToRORd function go
  **595 → 1011** (+70%) once deep linearization's CSE'd temporaries are
  added. For vectorized use over many cells (see
  `docs/vectorized_computations.md`) that is a real memory increase, and
  `cse=False` below is the way out of it.
- **CSE can now be turned off.** Both `generalized_rush_larsen` and
  `hybrid_rush_larsen` take a `cse` parameter, default `True`, exposed on the
  CLI as `--cse` / `--no-cse` for `ode2py`, `ode2c`, `ode2julia` and
  `ode2ufl`, and as `cse` under `[tool.gotranx]` in `pyproject.toml`.

  `True` runs one `sympy.cse` across every state being linearized at once, so
  a subexpression shared *between* states is computed once. `False` performs
  no CSE at all: every `d<state>_dt_linearized` becomes one fully inlined
  expression, which emits no temporaries whatsoever — the option that avoids
  the memory cost described above — at the price of far more arithmetic.

  Measured operation count for the linearization block relative to the plain
  rhs, and number of temporaries:

  | model | `cse=True` ops / temps | `cse=False` ops / temps |
  |---|---|---|
  | `fitzhughnagumo` | 0.51x / 2 | 0.62x / 0 |
  | `lorentz` | 0.38x / 0 | 0.38x / 0 |
  | `beeler_reuter_1977` | 0.34x / 7 | 0.59x / 0 |
  | `tentusscher_panfilov_2006_M_cell` | 0.67x / 74 | 1.38x / 0 |
  | `ORdmm_Land` | 1.23x / 381 | 12.65x / 0 |
  | `ToRORd_dyn_chloride` | 1.08x / 430 | 9.43x / 0 |

  Note that 1.8.0 already ran CSE, but one pass per state rather than one
  across all of them. That per-state strategy was briefly kept as a third
  option and then dropped: it never saved more than 11% of the temporaries
  and always cost 18–31% more operations than a joint pass, so no preference
  selected it. If a future change starts emitting `del` for a temporary once
  it is dead — which nothing does today, in any backend — per-state becomes
  worth reconsidering, since a per-state temporary is provably dead at the
  end of that state's own update where a shared one may not be.

  If you diff generated output across gotranx versions: regenerating an
  existing model now produces different temporary names than 1.8.0 (the
  shared pool is `_linearization_temp_<k>`, not `_<derivative>_linearized_<k>`),
  and the different factoring reassociates floating-point operations, which
  can shift results at the ~1e-14 level.
