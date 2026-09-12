from __future__ import annotations
from typing import Any
from pathlib import Path

import typer

from ..codegen import CodeGenerator
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


def find_pyproject_toml_config() -> Path | None:
    """Find the pyproject.toml file."""
    from black.files import find_pyproject_toml

    path = find_pyproject_toml((str(Path.cwd()),))
    if path is None:
        return None
    return Path(path)


def read_config(path: Path | None) -> dict[str, Any]:
    """Read the configuration file."""

    # If no path is given, try to find the pyproject.toml file
    if path is None:
        path = find_pyproject_toml_config()

    # Return empty dict if no path is found
    if path is None:
        return {}

    # Try to read the configuration file
    try:
        # First try to use tomllib which is part of stdlib
        import tomllib as toml
    except ImportError:
        # If tomllib is not available, try to use toml
        try:
            import toml  # type: ignore
        except ImportError:
            typer.echo("Please install 'tomllib' or 'toml' to read configuration files")
            return {}

    try:
        config = toml.loads(Path(path).read_text())
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


def validate_scheme(scheme: list[Scheme] | list[str]) -> list[Scheme]:
    lst = []
    for s in scheme:
        if isinstance(s, str):
            lst.append(Scheme(s))
        else:
            lst.append(s)
    return lst
