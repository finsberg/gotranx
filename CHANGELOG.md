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
