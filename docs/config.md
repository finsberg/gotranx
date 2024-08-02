---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Configuration

Using the command line you can see the available options using the `--help` flag. For example
```{code-cell} shell
!gotranx --help
```
or more specifically
```{code-cell} shell
!gotranx ode2py --help
```

## Specify configurations in `pyproject.toml`

It is also possible to specify the options in your `pyproject.toml`, e.g
```toml
# pyproject.toml

[tool.gotranx]
verbose = true
delta = 1e-6
scheme = [
    "explicit_euler",
    "generalized_rush_larsen",
    "hybrid_rush_larsen",
]
stiff_states = [
    "m",
    "h",
    "j",
]

[tool.gotranx.python]
format = "ruff"

[tool.gotranx.c]
format = "clang-format"
```

This will override any arguments passed from the command line. If you want to specify another configuration file, you can also pass the `--config` (or `-c`) flag where you specify a configuration file, e.g

```shell
gotranx ode2py file.ode -c folder/pyproject.toml
```

## Options


### General options (under `tool.gotranx`)

- `verbose` (boolean, default: `false`): If True display more logging
- `scheme` (list[str], default: []): Which schemes to include, see
```{code-cell} shell
!gotranx list-schemes
```
- `delta` (float, default: 1e-8): Tolerance for zero division check in Rush-Larsen schemes
- `stiff_states`: (list[str], default: []): List of states where to apply the Rush-Larsen scheme for Hybrid Rush Larsen

### Python specific options (under `tool.gotranx.python`)

- `format` (str, default: `black`). Formatter to use for the python code

```{code-cell} python
import gotranx

print(gotranx.codegen.PythonFormat._member_names_)
```

- `backend` (str, default: `numpy`). Backend to use for the python code

```{code-cell} python
import gotranx

print(gotranx.cli.gotran2py.Backend._member_names_)
```

### C specific options (under `tool.gotranx.c`)

- `format` (str, default: `clang-format`). Formatter to use for the C code

```{code-cell} python
import gotranx

print(gotranx.codegen.CFormat._member_names_)
```
- `to` (str, default `.h`). Whether to save the C code to a `.c` or `.h` file
