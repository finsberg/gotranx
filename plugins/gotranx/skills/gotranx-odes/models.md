# Where models come from

gotranx works on `.ode` files. Everything else is converted to `.ode` first (CellML) or loaded
directly (`gotranx.load_ode` reads `.ode`; `gotranx.compose` reads `.ode` and `.cellml` parts).

## The user's own `.ode` file

The grammar: https://finsberg.github.io/gotranx/docs/grammar.html (source:
[docs/grammar.md](https://github.com/finsberg/gotranx/blob/main/docs/grammar.md)). The essentials:

```
parameters("membrane",
    Cm = ScalarParam(1.0, unit="uF/cm**2", description="capacitance"),
    g_Na = 23.0
)
states("membrane", V = -84.0, m = 0.0, h = 1.0)

expressions("membrane")
i_Na = g_Na*m**3*h*(V - 50.0)
dV_dt = -i_Na/Cm
dm_dt = (m_inf - m)/tau_m
...
```

- every state `X` needs exactly one `dX_dt`; a name can be assigned only once;
- `Conditional(cond, a, b)`, `Min`, `Max` (capitalised) are functions; a variable called exactly
  `Min` or `Max` does not parse;
- a dot in a component name nests it (`"ep.gate"` is under `"ep"`).

Check that it loads before anything else:

```bash
python -c "import gotranx; ode = gotranx.load_ode('model.ode'); print(ode.num_states, 'states', ode.num_parameters, 'parameters')"
```

## CellML

```bash
gotranx cellml2ode model.cellml -o model.ode
```

The conversion goes through myokit. Read the resulting `.ode` before using it:

- **Stimulus.** CellML cell models usually contain their own periodic stimulus (parameters such as
  `stim_amplitude`, `stim_start`, `stim_period`, `IstimAmplitude`). For a single-cell run that is what
  you want — read when it starts: it differs between files (the ToR-ORd CellML in gotranx's tests
  starts at 0 ms, the O'Hara-Rudy 2011 CellML of the split-ode example at 50 ms). For a tissue
  simulation where the PDE solver stimulates, set the amplitude to zero, or the tissue is stimulated
  twice. Find the names with `grep -in stim model.ode`. Sign and start differ between models:
  ToR-ORd `i_Stim_Amplitude = -53` A/F from 0 ms; TP06 `stim_amplitude = +52` A/F from
  `stim_start = 10` ms.
- **Names and units** are those of the CellML file (often long, e.g. `membrane_V`). Note the name of
  the membrane potential state (`grep -n "_dt = " model.ode` lists the derivatives; ToR-ORd uses
  `v`) — downstream solvers ask for it (`v_name`, `v_index`).
- Export back with `gotranx ode2cellml model.ode -o model.cellml` (never singularity-guarded: it
  exports the model, not code).

## The odezoo registry

odezoo is a curated registry of published cell models (and some others), downloaded exactly as
recorded and verified by sha256. Use the **odezoo-models** skill if it is available; otherwise:

```bash
python -m pip install odezoo
odezoo search --category electrophysiology --species human
odezoo info tentusscher_2006_epi
odezoo fetch tentusscher_2006_epi -o models/
```

or from Python, `path = odezoo.get("land_2017", version="github-6e5a5aa")`. Pin `version=` in scripts
so a later registry update cannot change the model under you. Docs:
https://computationalphysiology.github.io/odezoo.

## Several models at once

Electrophysiology plus contraction (or any parts that exchange variables): see
[composition.md](composition.md).

## Which model for which purpose (starting points, confirm with the user)

- teaching / fast qualitative tissue runs: FitzHugh–Nagumo, Mitchell–Schaeffer, Beeler–Reuter;
- human ventricle: ten Tusscher–Panfilov 2006 (TP06, endo/mid/epi variants), O'Hara-Rudy 2011,
  ToR-ORd (endo/mid/epi; with dynamic chloride);
- atrium: Courtemanche 1998;
- contraction: Land 2017 (human, used with O'Hara-Rudy/ToR-ORd), Rice 2008 — see the
  crossbridge-models skill or https://computationalphysiology.github.io/crossbridge.
