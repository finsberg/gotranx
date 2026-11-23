# Composing models

`gotranx.compose` builds one model from several: a **recipe** (TOML) says which
models are combined and how each is adapted to the others, and a **glue file**
says how they are coupled. The result is an ordinary `gotranx.ODE`, in which each
part is one component named after the part. `ODE.split` then carves a component
such as `mechanics` out of the result by what goes in and what comes out.

The files below are the toy composition the tests use
(`tests/compose_files/`): a calcium model and a contraction model.

## The recipe

```toml
time_unit = "ms"                  # the composed model's clock
glue = "toy_glue.ode"

[[parts]]
name = "ep"                       # mandatory; the part's components become one
file = "toy_ep.ode"               # component "ep". .ode, or .cellml
time_unit = "ms"

[[parts]]
name = "toy"
file = "contraction_toy.ode"
time_unit = "s"                   # rescaled to the model's ms
rename = { dLambda = "dLambda_toy" }

[components.mechanics]            # optional: split at the end
inputs  = ["toy.B"]
outputs = ["toy.Zs", "toy.Ta"]
```

- `time_unit` (top level) and each part's `time_unit` are pint units of time, and
  both are mandatory.
- `rename` is optional. It renames a parameter, intermediate or state of that
  part everywhere it is defined or used, including a state's `d<X>_dt`.
- `allow_missing = true` lets the glue use names that no part defines.
- `[components.<name>]` is optional. Each table calls `ODE.split(name,
  inputs=..., outputs=...)` on the composed model, in recipe order. Names may be
  written `part.name`, which is checked against the part and then unqualified.
  If a split fails, `compose` raises a `CompositionError` whose messages each
  start `components.<name>:`.
- Recipes name local files only. Default values are not set here: calibration
  belongs where the model is used.

## The glue file

The glue file uses the `.ode` grammar plus three additions, accepted only in glue
files: `part.name` to reference another part's name, `replace`, and `+=` on a
state derivative.

```
# How ep and toy are coupled.
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
misuse is an error that names the fix, and all errors are reported together with
the glue file's line numbers.

| Statement, in a block for part X | Name exists in X? | Result |
|---|---|---|
| `name = ...` | no, and in no other part | new parameter or intermediate in X |
| `name = ...` | yes | error: already a parameter/intermediate of X: use it, or write `replace name = ...` |
| `name = ...` | no, but in another part | error: names must be unique; rename in the recipe or choose another |
| `replace name = ...` | yes | parameters block: a new declaration (default, unit, description). Expressions block: a parameter becomes an input, or an intermediate gets a new definition. |
| `replace name = ...` | no | error: "not a name of X" (catches typos) |
| `dX_dt += ...` | `X` is a state of X | the term is added to its derivative |
| `dX_dt += ...` | otherwise | error |
| `replace` on a state | | error for now |

A parameter of one component that another uses is written under both components,
so both halves of a split carry it. A state or intermediate used by another
component becomes a missing variable when the model is split.

A part whose `time_unit` differs from the model's has every state derivative
multiplied by `k`, one model time unit expressed in the part's unit, and `t`
replaced by `k*t`. Only derivatives and `t` change: an intermediate that is a
rate stays in the part's unit, so glue that reads one must convert it. The glue
itself is in the model's time unit.

## From Python

```python
import gotranx

ode = gotranx.compose("toy_recipe.toml")  # or a dict with the same keys
zeta = ode.split("mechanics", inputs=["B"], outputs=["Zs", "Ta"])
```

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
It prints, for each component, what it needs from the others:

```
ep needs: J
toy needs: cai
```

Those are the names the component would take as missing variables if every
other component were removed.

```{warning}
With `allow_missing = true` the model can be saved, but gotranx cannot load an
`.ode` file that uses names it does not define. `gotranx compose` still writes
the file, and prints a warning.
```
