# Composing models

`gotranx.compose` builds one model from several: a **recipe** (TOML) says which
models are combined and how each is adapted to the others, and a **glue file**
says how they are coupled. The result is an ordinary `gotranx.ODE`, in which each
part keeps its components, nested under the part's name. `ODE.split` then carves a
component such as `mechanics` out of the result by what goes in and what comes out.

The files below are the toy composition the tests use
(`tests/compose_files/`): a calcium model and a contraction model.

## The recipe

`toy_recipe.toml`:

```toml
# The toy composition of tests/test_compose.py; toy_coupled.ode is its result, by hand.
time_unit = "ms"
glue = "toy_glue.ode"

[[parts]]
name = "ep"
file = "toy_ep.ode"
time_unit = "ms"

[[parts]]
name = "toy"
file = "../odefiles/contraction_toy.ode"
time_unit = "s"
rename = { dLambda = "dLambda_toy" }  # the glue adds a parameter dLambda, as for Land
```

- `time_unit` (top level) is the composed model's clock, and each part's
  `time_unit` is the part's own. Both are pint units of time, and both are
  mandatory. Here `toy` is in seconds, and is rescaled to the model's
  milliseconds.
- Each part has a `name`, which must be a valid name, so without a dot. Its
  components keep their names under it: `toy_ep.ode` has two, `calcium` and
  `gate`, which become `ep.calcium` and `ep.gate`, and `contraction_toy.ode`'s
  `kinetics` becomes `toy.kinetics`; see "Components" below.
- `file` is an `.ode` or a `.cellml` file. Relative paths are relative to the
  recipe.
  On Windows, write paths with forward slashes, or in single quotes: inside
  double quotes, TOML reads a backslash as the start of an escape sequence.
- `rename` is optional. It renames a parameter, intermediate or state of that
  part everywhere it is defined or used, including a state's `d<X>_dt`.
- `allow_missing = true` lets the glue use names that no part defines.
- Recipes name local files only. Default values are not set here: calibration
  belongs where the model is used.

A recipe may also split the composed model, with one optional table per new
component:

```toml
[components.mechanics]
inputs  = ["toy.B"]
outputs = ["toy.Zs", "toy.Ta"]
```

Each table calls `ODE.split(name, inputs=..., outputs=...)` on the composed
model, in recipe order. Names may be written `part.name`, which is checked
against the part and then unqualified. If a split fails, `compose` raises a
`CompositionError` with every message of that split, each starting
`components.<name>:`.

## The glue file

The glue file uses the `.ode` grammar plus three additions, accepted only in glue
files: `part.name` to reference another part's name, `replace`, and `+=` on a
state derivative.

```
# How ep (toy_ep.ode) and toy (contraction_toy.ode) are coupled.
# In a block for part X, bare names are X's; another part's are written part.name.

parameters("toy",
lmbda = 1.0,
dLambda = 0.0
)

expressions("toy")
replace Ca = 1000.0*ep.cai
replace SL = lmbda*SL0
replace dSL = 1000.0*dLambda*SL0
J = 0.07*dCaTRPN_dt

expressions("ep")
dcai_dt += -toy.J
```

A plain statement creates a new name; `replace` changes an existing one. Every
misuse is an error that names the fix, with the glue file's line number.
`compose` raises one `CompositionError` with the errors it found: those in the
recipe, and every misuse in the table below. Some stop it before the rest are
checked:

- The glue file is parsed up to its first syntax error, unknown statement (a
  word other than `replace`), `states` block, or block that does not name
  exactly one part, so it reports one of those at a time.
- A part that cannot be found or loaded, or a time unit that is missing or not
  a unit of time, stops the glue from being checked.
- Circular equations are reported once everything else is right, one cycle at
  a time.
- The `[components]` tables are split in order, and the first split that fails
  is reported, with every message of that split; the tables after it are not
  tried.

| Statement, in a block for part X | Name exists in X? | Result |
|---|---|---|
| `name = ...` | no, and in no other part | new parameter or intermediate in X |
| `name = ...` | yes | error: already a parameter/intermediate of X: use it, or write `replace name = ...` |
| `name = ...` | no, but in another part | error: names must be unique; rename in the recipe or choose another |
| `replace name = ...` | yes | parameters block: a new declaration; the unit and description are kept unless it gives them. Expressions block: a parameter becomes an input, keeping its unit unless one is given, or an intermediate gets a new definition. |
| `replace name = ...` | no | error: "not a name of X" (catches typos) |
| `dX_dt += ...` | `X` is a state of X | the term is added to its derivative |
| `dX_dt += ...` | otherwise | error |
| `replace` on a state | | error for now |

A parameter of one part that another part uses is written under its own
component and under each component that uses it, here
`parameters("ep.calcium", "toy", k_up=...)` for a `J = ep.k_up*dCaTRPN_dt` in
the glue, so that both halves of a split carry it. A state or intermediate used
by another part becomes a missing variable when the model is split.

A part whose `time_unit` differs from the model's has every state derivative
multiplied by `k`, one model time unit expressed in the part's unit, and `t`
replaced by `k*t`. Only derivatives and `t` change: an intermediate that is a
rate stays in the part's unit, so glue that reads one must convert it. The glue
itself is in the model's time unit.

## Components

A dot in a component name nests it under every prefix that ends before a dot:
`ep.gate` is under `ep`, and `a.b.c` under `a.b` and `a`. `ode.get_component("ep")`
returns the atoms of `ep` and of every component nested under it, and
`ode - ode.get_component("ep")` removes them all. An atom that is also in a
component outside them stays. Each atom still lists the components it is in, so
this holds for any `.ode` file, and one without dots behaves as it always has.

In a composed model, part `ep`'s component `calcium` is `ep.calcium`, and its
unnamed component, if it has one, is `ep` itself. What the glue defines for
`ep` is in `ep` too, and a name the glue replaces stays where it was. The saved
model has blocks such as `states("ep.gate", ...)` and
`parameters("toy.kinetics", ...)`, and:

```
expressions("ep.calcium")
release = c_rel*n
dcai_dt = release + (-J - cai*k_up)

expressions("ep.gate")
dn_dt = (0.02*n)*((-1000)*cai) + 0.01*(1 - n)

expressions("toy")
J = 0.07*dCaTRPN_dt
```

`ODE.split` follows the outputs' dependencies within the parts they belong to,
across those parts' components, and what moves keeps its place under the new
name: with `ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])`,
`toy.crossbridges` becomes `mechanics.crossbridges` and `toy` becomes
`mechanics`. An intermediate computed on both sides, or a parameter used on
both, keeps its components and gains those under `mechanics`. What moves from
another part, such as a parameter of `ep` that only the new component uses, goes
to `mechanics` itself. A split cannot be named after a component, or after a
name that holds components, such as `ep`.

## From Python

```python
import gotranx

ode = gotranx.compose("toy_recipe.toml")  # or a dict with the same keys
split = ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
```

With the `[components.mechanics]` table above in the recipe, `compose` returns
the split model itself.

`compose` takes a path, or a dict with the same keys and `base_dir=` for its
relative paths. The saved model has a header with the recipe and the sha256 of
every file read, and no timestamp.

Like `load_ode`, `compose` guards the model's removable singularities by
default, which is what code generation wants. To save the model, compose it
without, as `gotranx compose` does:

```python
gotranx.compose("toy_recipe.toml", remove_singularities=False).save("toy.ode")
```

An `.ode` file is written unguarded, and every gotranx command that loads one
guards it. Saving a guarded model would have it guarded again on load: each
guard nested in a second one, and the time to find them spent twice.

## From the command line

```
gotranx compose toy_recipe.toml -o toy.ode
```

It writes the model, unguarded, by default to `<recipe stem>.ode` in the current
directory, and refuses to write over the recipe, the glue file or a part's file.
It prints, for each part and each split, what it needs from the others:

```
Wrote toy.ode
toy needs: cai
ep needs: J
```

Those are the names the part would take as missing variables if every other
part and split were removed, each with the components nested under it. There is
one line per part, not one per component such as `ep.calcium`.

```{warning}
With `allow_missing = true` the model can be saved, but gotranx cannot load an
`.ode` file that uses names it does not define. `gotranx compose` still writes
the file, and prints a warning.
```
