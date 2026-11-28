# Choosing a scheme and a time step

A generated scheme takes **one** explicit step: `scheme(states, t, dt, parameters)` returns the
states at `t + dt`. List what the installed version offers with `gotranx list-schemes`. Background:
https://finsberg.github.io/gotranx/docs/schemes/index.html and the second-order tutorial
https://finsberg.github.io/gotranx/examples/second-order-rush-larsen/main.html (source:
[examples/second-order-rush-larsen/main.py](https://github.com/finsberg/gotranx/blob/main/examples/second-order-rush-larsen/main.py)).

| Scheme (`--scheme`) | Order | Exponential update for | Cost per step |
|---|---|---|---|
| `explicit_euler` | 1 | no state | 1 rhs |
| `rush_larsen` | 1 | gating states, detected automatically | ~1 rhs + exps |
| `generalized_rush_larsen` | 1 | every state | ~1 rhs + exps + derivatives |
| `hybrid_rush_larsen` | 1 | the states you list (`--stiff-states`) | ~1 rhs + exps |
| `rush_larsen_2` | 2 | gating states, detected automatically | up to ~2× first order |
| `generalized_rush_larsen_2` | 2 | every state | up to ~2× first order |
| `hybrid_rush_larsen_2` | 2 | the states you list | up to ~2× first order |

## How to choose

- **Cell model in fenicsx-beat (or another operator-splitting tissue solver):** `rush_larsen_2` —
  fenicsx-beat's default since 0.14. Second order matters only together with Strang splitting and a
  second-order PDE step (θ = 0.5); with Godunov splitting a first-order scheme is enough.
- **Robustness first (large `dt`, stiff model, explicit `v` blows up):** `generalized_rush_larsen`.
  It is stable at the largest `dt`, but it can be visibly less accurate at the same `dt` (e.g. TP06
  + Land at 0.05 ms: action-potential peak 4.6 mV above the reference, against 0.5 mV for
  `rush_larsen_2`), and it exponentiates every state separately, so linear
  invariants (e.g. a Markov chain's occupancies summing to 1, conserved total charge or ion content)
  are not preserved exactly. Explicit schemes (`explicit_euler`, the explicit parts of RL/RL2) keep
  linear invariants for the states they step explicitly.
- **You know which states are stiff:** `hybrid_rush_larsen` / `hybrid_rush_larsen_2` with
  `--stiff-states m --stiff-states h ...` (the option is repeated per state). Without any
  stiff states, hybrid is plain forward Euler.
- **Cheapest, small `dt` anyway:** `explicit_euler` (stable only when `dt` is well below the
  fastest time scale; for ORdmm_Land it gives NaN during the upstroke at 0.05 ms and is stable at
  0.025 ms; FitzHugh–Nagumo-type models allow much larger steps).
- **Reference solution / validation:** generate only `rhs` and integrate with an adaptive stiff
  solver: `scipy.integrate.solve_ivp(lambda t, y: model.rhs(t, y, p), (0, T), y0, method="BDF",
  rtol=1e-8, atol=1e-10, max_step=0.5 * stim_duration, t_eval=times)` (`t_eval`: values at the
  stepper's times, for comparison). **Set `max_step`** to at most half the
  stimulus duration (and ≤ 0.25 ms for detailed models): otherwise the adaptive solver can step over
  a 1 ms stimulus and the cell never fires. Too slow for tissue, ideal for checking a scheme.

**See what "automatic" means for this model** before trusting `rush_larsen`/`rush_larsen_2`:

```python
import gotranx
from gotranx.linearization import gating_states

ode = gotranx.load_ode("model.ode")
print(gating_states(ode))  # states that get the exponential update
```

For ToR-ORd this includes the IKr Markov-chain states (`C1`, `C2`, `C3`, ...) and `cansr`, so their
sum/conservation is not exact. Check it if the user studies those states (track
`C1 + C2 + C3 + O_ + I_` over a beat at the chosen `dt`). Use `hybrid_rush_larsen_2` with an explicit list to exclude them.

## Choosing `dt`

1. Start from what the consumer uses (fenicsx-beat demos: 0.05 ms with `rush_larsen_2`; detailed
   models often need ≤ 0.05 ms around the upstroke).
2. Run one beat at `dt`, `dt/2` and `dt/4` (or against the BDF reference above) and compare what the
   user cares about — action-potential duration (APD90), peak, upstroke time, calcium peak. The
   first-order schemes should halve the error when `dt` halves, the `_2` schemes quarter it. If not,
   `dt` is outside the asymptotic range (or a discontinuity, e.g. the stimulus edge or a
   `Conditional`, dominates).
3. A larger *stable* `dt` is not a more accurate one: at a given `dt`, forward Euler can be more
   accurate than an exponential scheme when it is stable. Example (ORdmm_Land, one beat, against a
   BDF reference with peak 42.6 mV and APD90 337.3 ms): at `dt` = 0.1 ms `rush_larsen_2` gives
   43.1 mV / 336.9 ms, `generalized_rush_larsen` 50.9 mV / 334.6 ms, `explicit_euler` NaN.

## Options that change the generated scheme

- `--delta` (default 1e-8): the zero-division guard in the Rush–Larsen update
  (`|a| > delta`); leave it.
- `--cse/--no-cse` (default on): common-subexpression elimination across the linearized
  expressions. `--no-cse` only helps the vectorized numpy backend (no temporaries per array).
- `--remove-singularities/--no-remove-singularities` (default on): replaces a small
  window around each removable 0/0 (gate rates like `(V + 10)/(exp((V + 10)/10) - 1)`, GHK fluxes at
  `v = 0`) with a Taylor series. Keep it on: without it the linearization can be badly wrong near
  the pole and numba crashes on 0/0. Turn it off only to reproduce code generated by older versions.
- Several `--scheme` flags generate several scheme functions into one file.
