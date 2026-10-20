from __future__ import annotations
from typing import Any
from pathlib import Path

import typer

from ..codegen import CodeGenerator
from ..ode import ODE
from ..schemes import Scheme, get_scheme


def add_schemes(
    codegen: CodeGenerator,
    scheme: list[Scheme] | None = None,
    delta: float = 1e-8,
    stiff_states: list[str] | None = None,
    cse: bool = True,
) -> list[str]:
    comp = []
    if scheme is not None:
        for s in scheme:
            kwargs: dict[str, Any] = {}
            if "rush_larsen" in s.value:
                kwargs["delta"] = delta
                kwargs["cse"] = cse
            if s.value == "hybrid_rush_larsen":
                kwargs["stiff_states"] = stiff_states

            comp.append(codegen.scheme(get_scheme(s.value), **kwargs))
    return comp


def _toml_loads(text: str) -> dict[str, Any]:
    import tomllib

    return tomllib.loads(text)


def find_pyproject_toml_config(start: Path | None = None) -> Path | None:
    """The nearest `pyproject.toml` with a `[tool.gotranx]` table, searching from
    `start` (by default the current directory) up to the root of the file system.

    A `pyproject.toml` that cannot be parsed is returned too, so that
    `read_config` reports it rather than skipping it silently.
    """
    start = Path.cwd() if start is None else start
    for directory in (start, *start.parents):
        path = directory / "pyproject.toml"
        if not path.is_file():
            continue
        try:
            config = _toml_loads(path.read_text())
        except Exception:
            return path
        if "gotranx" in config.get("tool", {}):
            return path
    return None


def read_config(path: Path | None) -> dict[str, Any]:
    """Read the configuration file."""

    # If no path is given, try to find the pyproject.toml file
    if path is None:
        path = find_pyproject_toml_config()

    # Return empty dict if no path is found
    if path is None:
        return {}

    try:
        config = _toml_loads(Path(path).read_text())
    except Exception:
        typer.echo(f"Could not read configuration file {path}")
        return {}
    else:
        return config.get("tool", {}).get("gotranx", {})


def validate_cse(cse: Any) -> bool:
    """Reject a non-boolean `cse`, which can only reach here from a config file.

    `cse` used to name a strategy (`"joint"` / `"per_state"` / `"none"`).
    Every non-empty string is truthy, so a stale `cse = "none"` -- the
    spelling that used to ask for *no* CSE -- would otherwise silently turn
    CSE on, the exact opposite of what it says.
    """
    if not isinstance(cse, bool):
        raise typer.BadParameter(
            f"cse must be true or false, got {cse!r}. "
            "It was previously a strategy name; use true for the old "
            "'joint' or 'per_state', and false for the old 'none'.",
            param_hint="cse",
        )
    return cse


def validate_remove_singularities(remove_singularities: Any) -> bool:
    """Reject a non-boolean `remove_singularities`, which can only reach here
    from a config file -- where a string such as "false" is truthy and would
    silently do the opposite of what it says."""
    if not isinstance(remove_singularities, bool):
        raise typer.BadParameter(
            f"remove_singularities must be true or false, got {remove_singularities!r}",
            param_hint="remove_singularities",
        )
    return remove_singularities


def validate_scheme(scheme: list[Scheme] | list[str]) -> list[Scheme]:
    lst = []
    for s in scheme:
        if isinstance(s, str):
            lst.append(Scheme(s))
        else:
            lst.append(s)
    return lst


def select_components(ode: ODE, remove: list[str] | None = None, keep: str | None = None) -> ODE:
    """The part of `ode` to generate code for: without the components in `remove`
    (`ode - ode.get_component(name)` for each), or only the component `keep`
    (`ode.get_component(keep).to_ode()`). Names the parts share become missing
    variables of the result."""
    if remove and keep is not None:
        raise typer.BadParameter("give --remove-component or --keep-component, not both")
    names = [c.name for c in ode.components]
    for name in [*(remove or []), *([keep] if keep is not None else [])]:
        if name not in names:
            raise typer.BadParameter(
                f"no component {name!r} in {ode.name}; it has {', '.join(names)}",
                param_hint="--remove-component" if keep is None else "--keep-component",
            )
    if keep is not None:
        return ode.get_component(keep).to_ode()
    if not remove:
        return ode
    # `ode - component` is named "<ode> - <component>", which is not a valid default
    # model name (ode2cytozoo derives the struct name from it).
    name = f"{ode.name}_without_{'_'.join(remove)}"
    for comp in remove:
        ode = ode - ode.get_component(comp)
    return ODE(components=ode.components, t=ode.t, name=name, comments=ode.comments)
