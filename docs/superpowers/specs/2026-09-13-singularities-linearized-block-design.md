# Removable singularities in the Rush-Larsen linearized block

Design document. Status: approved, pending implementation plan.

## Problem

`gotranx` 2.0.0 changed both Rush-Larsen schemes to linearize on the true Jacobian
diagonal `∂f_i/∂y_i`, differentiating through intermediate assignments
(`src/gotranx/linearization.py`, `diagonal_jacobian`). That fixed a real bug in the
linearization, but differentiating through an intermediate turns a removable
singularity into a worse one: a gate rate `x/(exp(x) - 1)` has a simple removable
pole, and its derivative has a double pole. Before 2.0.0 these derivatives were never
formed, because the shallow derivative was zero and the state fell back to forward
Euler. Now every state gets one.

The existing machinery (`ODE.remove_singularities`, `Assignment.singularities`,
`atoms.remove_singularities`) runs over the ODE's own assignments, before and
independently of the derivatives the schemes generate. It does not cover the
linearized block, and where its guard does survive differentiation it produces a wrong
number.

### Evidence

Five findings from probing the current implementation. Each one constrains the design.

**F1. Both flagship models have a reachable removable pole at `v = 0`.**
ToRORd and ORdmm compute GHK fluxes as `vffrt*(...)/(exp(vfrt) - 1)`
(`tests/odefiles/ToRORd_dyn_chloride.ode:395,436-438`). Since `vfrt = v*F/(R*T)` and
`vffrt = v*F*F/(R*T)`, this is the `x/(exp(x) - 1)` shape with a removable pole at
`v = 0` — which every action potential crosses twice per beat. Evaluating
`d(dv_dt)/dv` for ToRORd with default parameters and initial states:

| `v` | float64 |
|---|---|
| -1e-06 | -0.04053606112451487 |
| -1e-09 | 30.09536192424742 |
| 0.0 | ZeroDivisionError |
| 1e-09 | -48.82704721532448 |
| 1e-06 | -0.04052418247488419 |

The true value is about -0.0405. A positive linearization is the dangerous case:
`dv_dt_linearized` feeds `exp(dv_dt_linearized*dt)`, so +30 turns a decaying
exponential into a growing one. ORdmm has the same shape (6.25 / -10.66 at ∓1e-9
against a true -0.0055).

**F2. `sympy.singularities` on the linearized entry cannot find it.**
It returns `EmptySet` for `v` on both ToRORd and ORdmm, despite the pole being real.
The deep entry is written in terms of the intermediate symbol `vfrt`, not `v`, so a
scan in `v` sees no dependence. This is the same mechanism as the known CSE issue,
where `singularities(1/(1 - x0), V)` is `EmptySet` because the pole is in `x0`.

Consequence: **scanning the linearized block is structurally blind to the most
important case.** A scan of the deep entries finds the non-physical concentration
poles (`Ca_i` at 0 or at `-K_buf_c`) and misses the physiological voltage ones.

**F3. The source-level scan sees the GHK poles and discards all 13 of them.**
Every one is classified infinite and dropped by `Singularity.is_infinite`
(`src/gotranx/atoms.py:231`):

```
INab       vfrt = 0  -> oo*sign(PNab*vffrt*(nai - nao))                  [INFINITE-skipped]
PhiCaL_ss  vfrt = 0  -> oo*sign(vffrt*(-cao*gamma_cao + cass*gamma_cass)) [INFINITE-skipped]
```

Because `vffrt` and `vfrt` are separate named intermediates, sympy holds `vffrt` fixed
as `vfrt -> 0`, so the numerator's vanishing factor is invisible and a removable pole
is misclassified as essential. Across all five bundled models, `remove_singularities()`
today guards 15 expressions and not one of them is a GHK pole. This is a pre-existing
defect in the shipped mechanism, not only in the linearized block.

**F4. The wrong guarded value is an order-0 artifact.** For the toy model

```
parameters(C=1.0)
states("Membrane", V=-80.0, n=0.1)
expressions("Membrane")
alpha_n = (V + 10)/(exp((V + 10)/10) - 1)
dn_dt = alpha_n*(1 - n) - 0.125*n
dV_dt = -alpha_n*(V + 77)/C
```

the true limit of `d(dV_dt)/dV` at `V = -10` is `47/(2*C)`, i.e. 23.5 at `C = 1`.
Replacing `alpha_n` by a truncated Taylor series of increasing order and
differentiating through the guard:

| order | replacement | `d(dV_dt)/dV` at `V = -10` |
|---|---|---|
| 0 | `10` | **-10** (today's behaviour) |
| 1 | `5 - V/2` | `47/2` |
| 2 | `V**2/120 - V/3 + 35/6` | `47/2` |
| 4 | `-V**4/720000 - ... + 419/72` | `47/2` |

The derivative of a constant is zero, which is the entire bug. A replacement of order
>= 1 makes differentiation through the guard correct, so `diagonal_jacobian` needs no
singularity awareness of its own.

**F5. `sympy.limit` silently returns wrong values on float-coefficient gate rates.**

| expression | `limit` | `series` | truth |
|---|---|---|---|
| `0.2*(V+23)/(1-exp(-0.04*(V+23)))` | **0** | 5.0 | 5 |
| `(1/5)*(V+23)/(1-exp(-(1/25)*(V+23)))` | 5 | 5 | 5 |
| `(V+10.0)/(exp((V+10.0)/10.0)-1)` | **0** | 10.0 | 10 |
| `(V+10)/(exp((V+10)/10)-1)` | 10 | 10 | 10 |

`.ode` files are written with float coefficients, so `Assignment.singularities`'s use
of `limit` (`src/gotranx/atoms.py:305`) can already emit wrong replacement values
today, independently of everything else.

**Supporting measurements.**

- `sympy.singularities` terminates everywhere: 8.75 s over ToRORd's 52 deep entries,
  0.28 s on ORdmm's 48, 8.64 s source-level on ToRORd, 0.65 s on ORdmm. Tractable, but
  a 3-4x hit on the current 2.84 s codegen, and the wrong tool per F2.
- The existing zero-division `Conditional` in `_rush_larsen_update`
  (`src/gotranx/schemes.py:228`) is emitted for both ToRORd/`v` and BR/`V`, but it
  tests `|linearized| > delta`. It catches `linearized ~ 0`, not `±inf` or a wrong-sign
  O(1) value, so it does not cover any of this. Because the new guard lives on the
  source assignment rather than on the linearized value, the two never stack.
- The CSE hoist reproduces on the pinned sympy 1.14: `x1 = 1/(1 - x0)` is emitted
  outside the `Piecewise` and is `1/0` at `V = -40`.

### Tolerance

For `f'(x) = d/dx[x/(exp(x) - 1)]`, the quantity the linearized block needs, relative
error of the direct float64 formula against a truncated series replacement at
half-window `delta`:

| `delta` | direct `f'` | series N=2/3 |
|---|---|---|
| 1e-2 | 2.17e-12 | 1.11e-8 |
| **1e-3** | **8.6e-11** | **1.11e-11** |
| 1e-4 | 8.65e-9 | 1.11e-14 |
| 1e-5 | 1.94e-6 | — |
| 1e-7 | 1.13e-2 | — |
| 1e-8 | **2.22** | — |

The crossover is at `delta ~ 1e-3` in the local variable. Order 3 costs the same as
order 2 because the Bernoulli number `B_3` is zero. This gives roughly 1e-11 uniform
relative error, against today's 222% at 1e-8 followed by NaN at 0.

## Decisions

Settled with the author before design:

1. **Scope.** Replace the mechanism rather than extend it. Guard source assignments in
   the state variable, with cone inlining, a tolerance window, and a Taylor
   replacement. Do not scan the linearized block — F2 shows it cannot work.
2. **Which singularities to guard.** Removability after inlining, a purely mathematical
   test. No physiological reachability heuristic, no user-declared ranges. The
   concentration poles that motivated the question came from the deep scan, which this
   design abandons.
3. **Default.** On by default, with `--no-remove-singularities` to opt out.
4. **CSE.** Guard after CSE, implemented by holding guarded subtrees opaque during
   `cse`.
5. **PR #68.** Superseded. This work builds independently on `main`.

## Design

### New module: `src/gotranx/singularities.py`

A sympy-level helper with no `Assignment` or lark dependency, so both the ODE layer and
the schemes can use it. This is the reusable piece that does not exist today;
`Assignment.singularities` cannot be reused as-is because it needs
`self.value.dependencies`, which comes from the lark tree.

```python
@dataclass(frozen=True)
class RemovablePole:
    var: sympy.Symbol         # variable the pole lives in
    value: sympy.Expr         # location of the pole
    replacement: sympy.Expr   # truncated Taylor series, order >= 1
    half_width: float         # delta

def removable_poles(expr, var, *, order=3) -> frozenset[RemovablePole]: ...
def guard(expr, poles) -> sympy.Expr: ...
```

`guard` emits `Piecewise((replacement, Abs(var - value) < half_width), (expr, True))`,
folded when there is more than one pole.

`sympy.limit` is not used anywhere in this module. Removability is decided by whether
`sympy.series(expr, var, value, order + 1).removeO()` is free of `oo`, `zoo` and `nan`.
That single call is both the classification and the replacement, replacing the
`is_infinite` check (F3) and the `limit` call (F5) together.

### Which variable the guard lives in

This is the subtle part. A pole's removability is only visible after inlining —
`vffrt = vfrt*F` is what makes INab's numerator vanish — but `diagonal_jacobian`
differentiates with respect to intermediate symbols, computing `diff(expr, vfrt)`. If
the guard condition and the series were written in `v` while AD differentiates with
respect to `vfrt`, the series branch would differentiate to zero and the chain rule
would break silently. That is a fresh instance of the F4 failure.

The rule is therefore: **detect the pole and expand the series in the state variable,
and rewrite the whole assignment — both branches — in that variable.** Then `v` is a
free symbol of the guarded expression, the AD sweep seeded at `v` has `tangent[v] = 1`,
and the chain rule composes correctly through the guard. This is what makes an
order >= 1 series sufficient on its own, with no changes to `diagonal_jacobian`.

The cost is that a guarded assignment stops sharing `vfrt` and `vffrt` with the rest of
the model. In ToRORd that is 5 GHK terms each re-expanding `exp(v*F/(R*T))`. The effect
on generated size is measured, not assumed; see Measurements owed.

### Candidate pre-filter

Only assignments whose expression has a denominator containing a state-dependent symbol
are scanned. A denominator that is a parameter or a literal cannot vanish for a
state-dependent reason.

This is what keeps the scan affordable. `dv_dt = (I_stim - (i_Na + i_s + i_x1 +
i_K1))/C` is skipped without inlining its 10 013-operation cone, because `C` is a
parameter. Without the filter, the source scan costs 8.64 s on ToRORd.

### Choosing delta

Per pole, from the series coefficients: the largest `delta` satisfying

```
|a_{N+1} * delta^(N+1)| <= tol * |a_0|
```

with `tol = 1e-12`, clamped to `[1e-8, 1e-1]`. The lower bound keeps the window wider
than the region where the direct formula is already catastrophically wrong (2.22
relative error at 1e-8, per the tolerance table); the upper bound keeps the series from
being used where the direct formula is more accurate than the truncation. If `a_{N+1}`
is zero, the first nonzero higher coefficient is used. For the canonical kernel this
lands at `delta ~ 1e-3` in the local variable, matching the crossover in the tolerance
table.

This is scale-free: it adapts to a gate rate written in `-0.1*(V + 47)` as readily as
one written in `(V + 10)/10`, where a fixed `delta` in the guarded variable's own units
would not.

### Interaction with CSE

Guarded `Piecewise` subtrees are substituted for placeholder symbols before
`sympy.cse`, and restored afterwards. Nothing from inside a guard can then be hoisted
out, so no temporary is singular at a guarded value.

The division inside the guard still executes under numpy's `where`, which evaluates
both branches. That is `where` semantics rather than a hoist, and it is the behaviour
the acceptance criterion is about: the criterion is that no temporary computed *outside*
a guard divides by zero at the guarded value.

A block-level guard (an `if`/`else` around the temporaries) would avoid even that, but
it is unavailable on the vectorized numpy backend, where a branch over an array must be
a `where`. It is therefore not pursued.

### Numeric verification of every replacement

Symbolic tools have silently produced wrong answers twice in this investigation
(`limit -> 0` in F5, `singularities -> EmptySet` in F2). Every replacement is
spot-checked at build time by evaluating the original expression and the replacement at
`value ± delta/2`, with parameters and other states at their default values, and
comparing. A disagreement beyond tolerance logs a warning and the guard is dropped
rather than emitted. Both observed failures would have been caught by this.

### Wiring

`remove_singularities` becomes default-on, with `--no-remove-singularities` to opt out,
threaded through `load_ode` / `get_code` and the `ode2*` CLI subcommands. Today it is
reachable only from Python (`tests/test_python_codegen.py:719` and
`examples/run-python/main_numba.py:22`), so no CLI user can reach the fix at all. The
numba example exists precisely because numba crashes on division by zero where numpy
only warns.

## Scope note

Fixing F3 necessarily changes `rhs` output for ToRORd and ORdmm, not only the
linearized block: those 13 GHK poles are unguarded in the right-hand side today. This
is a larger diff than "cover the linearized block", and it was accepted deliberately —
guarding the linearized block while leaving the same pole unguarded in `rhs` would be
incoherent.

## Test plan

Mapped to the acceptance criteria.

1. **Beeler-Reuter `d(dV_dt)/dV` near `V = -23`.** Finite at exactly -23.0, and agreeing
   with a high-precision reference to ~1e-11 relative across the neighbourhood
   (-22.9, -22.999, -22.99999, -22.9999999, -23.0, and the mirror points), not only at
   the exact point.
2. **ToRORd and ORdmm `d(dv_dt)/dv` near `v = 0`.** The F1 table, re-run: finite at 0.0,
   correct sign and magnitude at ∓1e-9. This is a new criterion, from F1.
3. **The toy model emits a branch worth 23.5.** Written as a numeric assertion that the
   guarded branch evaluates to `47/2` at `V = -10`, not a grep for the literal `23.5` —
   the emitted branch is a polynomial, not a constant.
4. **CSE survival.** For every bundled model, evaluate every emitted CSE temporary at
   each guarded value and assert finiteness. The `Piecewise((0, V < -40), ((V+40)/(exp(-(V+40)/10)-1), True))`
   case is included as a direct regression test.
5. **No regression in the 2.0.0 criteria.** Every state of every bundled model still
   linearized; generated derivatives still agree with central differences through
   `rhs`; linearization block still within 2x the `rhs`
   (`test_linearization_block_stays_within_twice_the_rhs`).

   The stability criterion as stated — `base_model_IM.ode` stable at `dt = 0.02` ms
   over 300 ms — **cannot be run against this repo**: `base_model_IM.ode` is not a
   bundled fixture, and `tests/test_deep_linearization_models.py:299` says so
   explicitly, citing it as a design-doc reference measurement only. Substituted, as
   the closest runnable equivalent, by the existing stability tests over the bundled
   models at `dt = 1e-3` ms / 2000 steps
   (`tests/test_deep_linearization_models.py:379,447`), plus a new stability run at
   `dt = 0.02` ms over 300 ms on `ToRORd_dyn_chloride` and
   `tentusscher_panfilov_2006_M_cell`, which is where the F1 pole at `v = 0` actually
   lives. If the author can supply `base_model_IM.ode`, the criterion is run as
   originally written and this substitution is dropped.
6. **Full suite.** `python3 -m pytest` and `pre-commit run --all`.

Tests are written before the fix, and the failing output is shown before the
implementation lands.

## Measurements owed

Reported before and after, as requested:

- Codegen time for `ToRORd_dyn_chloride`, against the 2.84 s baseline.
- Generated source size for `ToRORd_dyn_chloride`, which the loss of `vfrt`/`vffrt`
  sharing in guarded assignments may increase.
- Linearization block operation count as a multiple of `rhs`, to confirm the 2x bound
  still holds.

## Out of scope

- Scanning the linearized block for singularities (F2: structurally cannot work).
- Block-level `if`/`else` guards in generated code (unavailable on the vectorized
  backend).
- Physiological range declarations in the `.ode` grammar.
- Lookup tables, and the other roadmap items.
