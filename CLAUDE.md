# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`gotranx` ("General ODE translator") is a Python library and CLI that parses a domain-specific
`.ode` markup language describing systems of ordinary differential equations, builds a symbolic
(`sympy`) representation, and generates source code for solving the ODE in several target
languages (Python, C, Julia, ModelingToolkit/Julia, UFL, Markdown docs). It can also import models
from CellML (via `myokit`).

## Development setup

```
python3 -m pip install -e ".[test]"     # editable install with test deps
python3 -m pip install -e ".[dev]"      # dev tools (pre-commit, ipython, pdbpp, ...)
python3 -m pip install -e ".[docs]"     # docs build deps
pre-commit install                      # install git hooks (ruff, mypy, cspell, ...)
```

Python >= 3.9 is required (CI runs 3.9–3.13 on Linux/macOS/Windows).

## Common commands

Run the full test suite:
```
python3 -m pytest
```

Run a single test file / test:
```
python3 -m pytest tests/test_ode.py
python3 -m pytest tests/test_ode.py::test_name -v
```

With coverage (as CI does):
```
python3 -m pytest --cov=gotranx --cov-report term-missing
```

Benchmarks (marked `benchmark`, uses `pytest-codspeed`/`pytest-benchmark`, excluded from a plain run
only if you filter them out — otherwise they run as normal tests):
```
python3 -m pytest tests/ --codspeed
```

Lint / format / type-check (same tools run by pre-commit):
```
pre-commit run --all           # ruff (lint+format), mypy, cspell, misc hygiene hooks
ruff check src tests
ruff format src tests
mypy                            # config lives in [tool.mypy] in pyproject.toml
```

Build the documentation (Jupyter Book):
```
python3 -m pip install -e ".[docs]"
jupyter-book build .
jupyter-book build -W --keep-going .   # warnings as errors, as used for gating docs changes
```

Run the CLI locally:
```
gotranx --help
gotranx ode2py file.ode --scheme explicit_euler -o file.py
gotranx ode2c file.ode -o file.c
gotranx ode2julia file.ode -o file.jl
gotranx ode2md file.ode -o file.md
gotranx cellml2ode model.cellml -o model.ode
gotranx list-schemes
```

## Architecture

The pipeline for every code-generation command is: **parse `.ode` text → build symbolic `ODE`
object → generate code via a `CodeGenerator` + `Template`**.

1. **Grammar & parsing** (`src/gotranx/ode.lark`, `parser.py`): a `lark` grammar defines the `.ode`
   DSL (`parameters(...)`, `states(...)`, `expressions(...)`/`component(...)`, assignments,
   math/logical functions). `Parser` (subclass of `lark.Lark`) loads this grammar. Parsing is LALR
   with `propagate_positions=True` so errors can be traced back to source locations.

2. **Transform to atoms** (`transformer.py`): `TreeToODE` walks the Lark parse tree and produces
   `LarkODE`, a collection of low-level `atoms.Atom` objects (`atoms.py` — `Parameter`, `State`,
   `Intermediate`, `StateDerivative`, etc., built with `attrs` and wrapping `sympy` symbols).

3. **Building the ODE model** (`ode_component.py`, `ode.py`): atoms are grouped into
   `ode_component.Component` objects, then `ode.make_ode()` combines components into the top-level
   `ode.ODE`. This step validates the model (`check_components` — every state must have a matching
   `d{name}_dt` derivative, no duplicate symbol definitions) and computes a dependency graph
   (`graphlib.TopologicalSorter`) used to topologically sort assignments and to support
   `remove_unused` (only emit expressions the state derivatives actually depend on).

4. **Entry point** (`load.py`): `load_ode(path)` / `ode_from_string(text)` run steps 1–3 and return
   an `ODE`. This is the main library API (`gotranx.load_ode`).

5. **Numerical schemes** (`schemes.py`): given an `ODE`, generates the update equations for a
   numerical integration scheme (explicit/generalized/hybrid Rush-Larsen, forward/backward Euler,
   etc. — see `gotranx list-schemes`). Schemes are plain functions with a `scheme_func` signature
   consumed by `codegen.base.CodeGenerator.scheme`.

6. **Code generation** (`codegen/`): `codegen/base.py` defines the abstract `CodeGenerator`, which
   assembles pieces (state/parameter index maps, initial value arrays, `rhs()`, `monitor_values()`,
   `missing_values()`, `scheme()`) by iterating `ode.sorted_assignments()` and printing each
   `sympy` expression via a language-specific `CodePrinter`. Each target language subclasses this
   in `codegen/{python,c,julia,mtk,ufl,jax,markdown}.py` and pairs with a `templates/{lang}.py`
   module (string templates for function signatures/boilerplate). Adding a new backend means adding
   a `codegen/<lang>.py` + `templates/<lang>.py` pair and a `Format`/printer as needed.

7. **CLI** (`cli/`): `cli/__init__.py` wires up a `typer` app with one subcommand per target
   (`ode2py`, `ode2c`, `ode2julia`, `ode2mtk`, `ode2ufl`, `ode2md`, `cellml2ode`, `list-schemes`,
   the deprecated `convert`). Each subcommand delegates to a matching `cli/gotran2<lang>.py` module
   that loads the ODE, calls into `codegen`, and writes output. `cli/utils.py` handles reading
   defaults from a `config.toml` (`[tool.gotranx]` section — see repo-root `config.toml` for an
   example; CLI flags override config file values).

8. **Units & sympy helpers**: `units.py` wraps `pint` for parsing/validating `ScalarParam(...,
   unit=...)` declarations; `sympytools.py` holds shared `sympy` utilities (e.g. symbol/expression
   manipulation used across codegen and schemes).

9. **CellML import** (`myokit.py`, `cli/cellml2ode.py`): converts CellML models to `.ode` text using
   `myokit`'s CellML importer, then formats/writes it out.

### Key invariants enforced by the model layer
- Every declared state must have exactly one corresponding `d{state}_dt` assignment across all
  components, or `ODE` construction raises `exceptions.ComponentNotCompleteError`.
- A symbol (parameter, state, or intermediate) can only be assigned once — reassignment raises an
  error during parsing/transformation.
- `remove_unused` support depends on the dependency graph built in `ode.py`; codegen filters
  emitted assignments through `self._condition` based on `ode.dependents()`.

## Docs & examples
- `docs/grammar.md` documents the `.ode` DSL in detail (operators, `ScalarParam`, components,
  conditionals, etc.) — read it before extending the grammar in `ode.lark`.
- `docs/cli.md` and `docs/config.md` document CLI usage and the `config.toml` format.
- `demo/` and `examples/` contain real-world `.ode`/`.cellml` models used in docs and as test
  fixtures (see `tests/test_demos.py`, `tests/odefiles`, `tests/cellml_files`, `tests/mmt_files`).

## Repo layout notes
- `sandbox/` is scratch/experimental content, not part of the package or docs build.
- `demo/` has its own `pyproject.toml` and is not part of the main package's dependency tree.
