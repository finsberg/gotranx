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

odezoo is a curated registry of published models, with a Python API and a CLI. Most of its models
are CellML electrophysiology models. Contraction models (Land 2017, Rice 2008, …) and a
circulation model are `.ode` files. Each file is downloaded exactly as recorded and checked
against its sha256. Use the **odezoo-models** skill if it is available. Otherwise, the docs (a page
per model, with search) are at https://computationalphysiology.github.io/odezoo, source
https://github.com/ComputationalPhysiology/odezoo.

**Install:** `python -m pip install odezoo` (Python 3.11 or later; ask first). Each release bundles
the registry as it was at release time. For models added since, install from GitHub
(`python -m pip install git+https://github.com/ComputationalPhysiology/odezoo`).

**Find and fetch:**

```bash
odezoo search --category electrophysiology --species human   # also free text: odezoo search tomek
odezoo values species                # each species and its number of models; also category, family, format
odezoo info tentusscher_2006_epi     # paper, DOI, cell type, versions, licence, upstream source
odezoo fetch tentusscher_2006_epi land_2017 -o models/   # writes models/<id>.cellml or models/<id>.ode
```

`odezoo search --format ode` lists the models that have an `.ode` version, and `odezoo search --ids`
prints only the ids (for scripts). From Python:

```python
import gotranx
import odezoo

odezoo.search("ten tusscher", category="electrophysiology", format="cellml")
path = odezoo.get("land_2017", version="github-6e5a5aa")  # the file in odezoo's cache
ode = gotranx.load_ode(path)  # .ode versions load directly
odezoo.get("torord_dyn_cl", dest="models")  # copied to models/torord_dyn_cl.cellml
```

- **CellML versions** need `gotranx cellml2ode` on the fetched file first (section above), except in
  a composition recipe, which takes the `.cellml` path directly ([composition.md](composition.md)).
  Pass `format="ode"` (`odezoo fetch <id> --format ode`) to get the `.ode` version of a model that has both.
- **Pin the version** in scripts and recipes: `odezoo info <id>` lists the version ids, then use
  `version=` in Python or `odezoo fetch <id> --version <version>`. Without it you get the default version, or
  the newest, and a later registry release could change the model under you.
- **Do not edit the file `odezoo.get` returns.** It is the cache entry, and the next `get` finds the
  sha256 wrong and silently puts the recorded file back. Copy it with `dest=` first, or better, keep
  the file unchanged and set parameters at run time or in a glue file.
- **Licence:** every file keeps its own (ToR-ORd: GPL-3.0-only, TP06: CC-BY-3.0, Land 2017: MIT).
  `odezoo info` shows it. Tell the user when they plan to redistribute the model or code generated
  from it.
- **Cache and offline use:** files already fetched are read from the cache, so they work offline.
  `odezoo cache dir` shows where it is (`ODEZOO_CACHE_DIR` moves it). `ODEZOO_REGISTRY` (or
  `odezoo --registry PATH`) points at a checkout of the odezoo repository, to use models not yet released.

Ids for the models this skill mentions:

| Model | odezoo id | Notes |
|---|---|---|
| ToR-ORd (dynamic chloride) | `torord_dyn_cl` | CellML, **endocardial**. gotranx's tests (and the numbers in this skill) use the mid-myocardial file, which differs only in `celltype` (0 endo, 1 epi, 2 mid) and the initial states. For mid or epi, set `celltype` at run time and pace to steady state first. |
| O'Hara-Rudy 2011 | `ohara_rudy_2011` | CellML, endocardial, with a `celltype` parameter. Its stimulus is **one pulse** at 50 ms (`amp`, `duration`, `stimStart`), with no period: pacing many beats needs a periodic stimulus added to the model, or one from the tissue solver |
| ten Tusscher–Panfilov 2006 | `tentusscher_2006_endo`, `tentusscher_2006_epi`, `tentusscher_2006_m` | CellML; epi is the file in gotranx's TP06 + Land test recipe |
| Land 2017 | `land_2017` | `.ode`, time in **seconds**; the file in gotranx's composition tests |
| Rice 2008 | `rice_2008` | `.ode` |
| Beeler–Reuter, Courtemanche, FitzHugh–Nagumo, Grandi 2010, T-World | `beeler_reuter_1977`, `courtemanche_1998`, `fitzhugh_nagumo_1961`, `grandi_2010`, `tworld` | CellML |

A model that `odezoo search` does not find (at the time of writing ORdmm_Land, Mitchell–Schaeffer,
and the mid/epi ToR-ORd files) has to come from the user, the authors' repository, or the Physiome
Model Repository.

## Several models at once

Electrophysiology plus contraction (or any parts that exchange variables): see
[composition.md](composition.md).

## Which model for which purpose (starting points, confirm with the user; odezoo ids above)

- teaching / fast qualitative tissue runs: FitzHugh–Nagumo, Mitchell–Schaeffer, Beeler–Reuter;
- human ventricle: ten Tusscher–Panfilov 2006 (TP06, endo/mid/epi variants), O'Hara-Rudy 2011,
  ToR-ORd (endo/mid/epi; with dynamic chloride);
- atrium: Courtemanche 1998;
- contraction: Land 2017 (human, used with O'Hara-Rudy/ToR-ORd), Rice 2008 — see the
  crossbridge-models skill or https://computationalphysiology.github.io/crossbridge.
