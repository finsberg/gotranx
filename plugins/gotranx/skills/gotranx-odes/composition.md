# Coupling two or more models (composition)

Typical case: an electrophysiology (EP) cell model and a contraction (crossbridge) model, e.g.
O'Hara-Rudy or ToR-ORd + Land 2017. gotranx (2.11 or later) composes them from a **recipe** (which parts, their
time units, renames) and a **glue file** (how the parts are coupled), and can **split** the result
again into halves that exchange "missing variables".

Read first: the how-to guide https://finsberg.github.io/gotranx/docs/composition.html (source:
[docs/composition.md](https://github.com/finsberg/gotranx/blob/main/docs/composition.md)) and the
worked example https://finsberg.github.io/gotranx/examples/split-ode/main.html (source:
[examples/split-ode/main.py](https://github.com/finsberg/gotranx/blob/main/examples/split-ode/main.py),
glue file [ord_land.glue.ode](https://github.com/finsberg/gotranx/blob/main/examples/split-ode/ord_land.glue.ode)).
Real recipes used by the tests:
[tests/compose_files](https://github.com/finsberg/gotranx/tree/main/tests/compose_files)
(ToR-ORd + Land, TP06 + Land).

## Interview (ask one at a time, only what is not yet known)

1. **Which parts?** File and format of each (`.ode` or `.cellml`), and where they come from
   (odezoo versions to pin).
2. **Coupling direction.**
   - *One-way*: EP calcium drives contraction; contraction does not feed back. Simplest; fine when
     the question is about tension.
   - *Two-way*: the contraction model's troponin binding also enters the EP calcium balance
     (Margara et al. 2021). Needed when contraction should affect the calcium transient or the
     action potential. The EP model's own troponin buffering must then be **replaced**, not kept,
     or troponin is counted twice. That needs an EP model with a *separate* troponin buffer term
     (O'Hara-Rudy, ToR-ORd: `trpnmax` inside `Bcai`). Models with one lumped cytosolic buffer
     (TP06: `Buf_c`, `K_buf_c`) have no troponin term to replace — offer one-way coupling (what
     gotranx's own TP06 + Land test recipe does), or a deliberate model change agreed with the user.
3. **Which variables cross?** Usually `Ca` (contraction input ← EP `cai`), the troponin flux
   (EP ← contraction), stretch `lmbda` and stretch rate `dLambda` (contraction ← mechanics or fixed).
   Ask the user to confirm names in both files; open the files and list candidates (TP06: `Ca_i`,
   membrane potential `V`; O'Hara-Rudy/ToR-ORd: `cai`, `v`). Land 2017 takes its inputs `Ca`, `SL`,
   `dSL` from a component `inputs`, which the glue replaces.
4. **Units of each part.** Time (ms vs s), concentration (mM vs µM), tension (kPa vs Pa). Land 2017
   is in s, µM and Pa; O'Hara-Rudy/ToR-ORd in ms and mM.
5. **One model or two halves?** One composed model is simplest. Split halves are needed when the
   parts are stepped differently (contraction less often), or the contraction part runs elsewhere
   (e.g. at quadrature points of a mechanics solver in simcardemsx).
6. **Who provides stretch?** Fixed (`lmbda = 1`, `dLambda = 0`: isometric single cell) or a
   mechanics solver that sets them every step.
7. **Parameters.** Contraction-model defaults may describe another preparation (Land's defaults are
   a skinned cell; the split-ode example sets intact human-cell values). Which set do they want?

## Alternatives to propose

**A. Compose into one model (recommended for single-cell work).** One `.ode`, one scheme, one `dt`;
the coupling is exact (no lag). Cost: the whole model is stepped at the EP time step.

**B. Compose, then split (`ODE.split`) into EP and contraction halves.** Same physics, but each half
is generated separately with `missing_values=` from the other; you exchange missing variables every
step (or step contraction only when its input changed, as the split-ode example does with a
tolerance). Needed for different time steps or when contraction lives inside a FE solver. Cost:
you write the exchange loop, initial missing values have a chicken-and-egg problem (compute the
inputs from the other half's initial state first), and coupling becomes explicit/lagged.

**C. Keep separate models and couple by hand.** Generate each model alone and pass values in your
own code. Not recommended: no name or unit checks, easy to double-count troponin; only for a quick
one-way experiment with an already-generated contraction model.

Orthogonal choice — **recipe as TOML + `gotranx compose`** (reproducible, the written `.ode` has a
header with the recipe and every input file's sha256; good for sharing and for downstream packages
that read `.ode` files) vs **recipe as a Python dict + `gotranx.compose`** (paths from
`odezoo.get(...)`, no files written; good inside a script).

## Building

Recipe (`ep_land.toml`):

```toml
time_unit = "ms"           # the composed model's clock (mandatory)
glue = "ep_land.glue.ode"

[[parts]]
name = "ep"
file = "ToRORd_dynCl_mid.cellml"
time_unit = "ms"           # each part's own clock (mandatory)

[[parts]]
name = "land"
file = "land_2017.ode"
time_unit = "s"
rename = { a = "a_land", h = "h_land", dLambda = "dLambda_land" }   # names that collide (see below)

# optional: split while composing
# [components.mechanics]
# inputs  = ["cai"]
# outputs = ["Ta", "J_TRPN"]
```

Glue file (`ep_land.glue.ode`), two-way coupling as in the tests' ToR-ORd + Land recipe:

```
parameters("land",
lmbda = 1.0,
dLambda = ScalarParam(0.0, unit="1/ms")
)

expressions("land")
replace Ca = 1000.0*ep.cai            # mM -> µM
replace SL = lmbda*SL0
replace dSL = 1000.0*dLambda*SL0      # 1/ms -> 1/s
J_TRPN = ep.trpnmax*dCaTRPN_dt

expressions("ep")
replace Bcai = 1.0/(1.0 + cmdnmax*kmcmdn/(cai + kmcmdn)**2)   # drop EP's own troponin buffering
dcai_dt += -Bcai*land.J_TRPN
```

This two-way glue uses ORd/ToR-ORd names (`cai`, `trpnmax`, `Bcai`, `cmdnmax`, `kmcmdn`); it does not
transfer to other EP models. For one-way coupling (any EP model), keep only the `replace Ca = ...`
line, with the EP model's calcium name, and the stretch lines, and leave the EP block out — e.g. for
TP06 `replace Ca = 1000.0*ep.Ca_i`.

**Renames.** Rename whatever the composition error names. With Land: `h` collides with the EP
model's `h` gate (TP06, ORd), `a` with an O'Hara-Rudy/ToR-ORd parameter (not needed for TP06), and
`dLambda` collides with the glue's own new `dLambda` parameter (Land already has an intermediate
of that name), so it is renamed even though no EP model defines it.

Glue rules: in a block for part X, bare names are X's and another part's are `part.name`; a plain
`name = ...` creates a new name, `replace name = ...` changes an existing one, `dX_dt += ...` adds a
term to a state's derivative. Errors name the fix and the glue line.

Then:

```bash
gotranx compose ep_land.toml -o ep_land.ode     # prints what each part needs from the others
gotranx ode2py ep_land.ode --scheme generalized_rush_larsen -o ep_land.py
```

or in Python:

```python
import gotranx

ode = gotranx.compose("ep_land.toml")
mech = ode.split("mechanics", inputs=["cai"], outputs=["Ta", "J_TRPN"])  # alternative B
mechanics = mech.get_component("mechanics").to_ode()
ep = mech - mech.get_component("mechanics")
code_ep = gotranx.cli.gotran2py.get_code(
    ep,
    scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    missing_values=mechanics.missing_variables,
)
code_mech = gotranx.cli.gotran2py.get_code(
    mechanics,
    scheme=[gotranx.schemes.Scheme.generalized_rush_larsen],
    missing_values=ep.missing_variables,
)
```

A half **that has missing variables** takes them as an extra last argument
(`scheme(states, t, dt, parameters, missing_variables)`, `missing_values(t, states, parameters,
missing_variables)`); a half without any (the EP half of a one-way coupling) has the ordinary
signatures (`scheme(states, t, dt, parameters)`, `missing_values(t, states, parameters)`). Check with
`inspect.signature`. `missing_values` computes what the *other* half needs. `ode.missing_variables`
is a dict `{name: index}` (index into the missing-variables array).

Choose the split's outputs for what the consumer needs: with `outputs=["Ta"]`, Land's passive state
`Cd` stays in the EP half; add `Tp` (passive tension) to the outputs if the mechanics side needs it.

## Checking

- `gotranx compose` succeeds and its "needs" lines are what you expect (contraction needs `cai`,
  EP needs the troponin flux; for one-way, EP needs nothing).
- One beat of the composed model: the calcium transient and action potential are close to the EP
  model alone (one-way: identical; two-way: slightly changed), active tension `Ta` rises after the
  calcium peak and returns to baseline. **Set Land's intact-cell parameters first** (the
  `land_parameters` dict and initial states `B = 1`, `CaTRPN = 0` in
  [examples/split-ode/main.py](https://github.com/finsberg/gotranx/blob/main/examples/split-ode/main.py)):
  with them peak tension is tens of kPa (≈ 60 kPa with TP06, one-way); with Land's skinned-cell
  defaults it is below 1 kPa. With intact parameters, values off by 1000 point to a unit conversion
  missing in the glue.
- For alternative B, compare the split halves against the composed model at the same `dt` (the
  split-ode example does this).

## Composition gotchas

- **Only derivatives and `t` are rescaled.** A part's `time_unit` multiplies its state derivatives
  and rescales `t`; an intermediate that is a *rate* stays in the part's unit, and concentrations
  and tensions are never converted. The glue is in the model's units: convert explicitly
  (`1000.0*ep.cai`, `1000.0*dLambda*SL0`).
- **Name collisions** between parts (Land's `a`, `h`, `dLambda` vs O'Hara-Rudy) must be renamed in
  the recipe; the error says which ("names must be unique: h is defined in ep and in land", "dLambda
  is already an intermediate of land").
- **Two-way coupling needs a separable troponin buffer** in the EP model; TP06's lumped buffer has
  none (one-way only, unless the model is changed deliberately).
- **Double-counted troponin** in two-way coupling unless the EP buffering term is replaced.
- **Skinned-cell defaults.** Land's default `ca50_ref` (2.5 µM) and `Tref` (40.5 kPa) are for a
  skinned cell; for an intact human cell use the values in the split-ode example.
- **Parameter and state overrides per half.** After a split, each half has only its own parameters
  and states; `init_parameter_values(**overrides)` / `init_state_values(**overrides)` raise
  `KeyError` for a name the half does not have — filter the overrides per half
  (`k in module["parameter"]`, `k in module["state"]`).
- **Save unguarded.** `gotranx compose` writes the model without singularity guards, which loading
  adds. In Python, save with `gotranx.compose(recipe, remove_singularities=False).save("model.ode")`
  — saving a guarded model nests the guards when it is loaded again.
- **`allow_missing = true`** lets the glue use undefined names and still write the file, but gotranx
  cannot load that `.ode` afterwards.
- **`replace` on a state** is not supported (error); couple through the derivative with `+=`.
- **Split names** cannot be the name of an existing component (e.g. `ep`).
- **Paths in TOML on Windows**: forward slashes or single quotes.
