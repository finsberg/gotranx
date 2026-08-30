"""Inspect an ODE file from the command line."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..load import load_ode


def _format_optional(value) -> str:
    if value is None:
        return "-"
    return str(value)


def _format_components(components) -> str:
    if components is None:
        return "-"
    if isinstance(components, (tuple, list)):
        return ", ".join(str(component) for component in components) or "-"
    return str(components)


def inspect(
    ode_file: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        help="Path to the .ode file to inspect.",
    ),
    list_states: bool = typer.Option(
        False,
        "--list-states",
        is_flag=True,
        help="List all states with their default values.",
    ),
) -> None:
    """Print basic information about an .ode file."""
    console = Console()
    try:
        ode = load_ode(str(ode_file))
    except Exception as exc:
        console.print(f"[red]Error loading ODE:[/red] {exc}")
        raise typer.Exit(1)

    states = getattr(ode, "states", []) or []
    parameters = getattr(ode, "parameters", []) or []
    state_derivatives = getattr(ode, "state_derivatives", []) or []
    intermediates = getattr(ode, "intermediates", []) or []
    missing_variables = getattr(ode, "missing_variables", []) or []

    console.print(f"[bold cyan]Inspecting {ode_file}[/bold cyan]")

    summary = Table(title="Summary")
    summary.add_column("Quantity", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Name", _format_optional(getattr(ode, "name", None)))
    summary.add_row("States", str(len(states)))
    summary.add_row("Parameters", str(len(parameters)))
    summary.add_row("State derivatives", str(len(state_derivatives)))
    summary.add_row("Intermediates", str(len(intermediates)))
    summary.add_row("Missing variables", str(len(missing_variables)))
    console.print(summary)

    if list_states:
        states_table = Table(title="States")
        states_table.add_column("Name", style="cyan")
        states_table.add_column("Value", justify="right")
        states_table.add_column("Unit")
        states_table.add_column("Components")
        states_table.add_column("Description")
        for state in states:
            name = getattr(state, "name", "?")
            value = getattr(state, "value", None)
            unit_str = getattr(state, "unit_str", None)
            if unit_str is None:
                unit_str = getattr(state, "unit", None)
            description = getattr(state, "description", None)
            components = getattr(state, "components", None)
            states_table.add_row(
                str(name),
                _format_optional(value),
                _format_optional(unit_str),
                _format_components(components),
                _format_optional(description),
            )
        console.print(states_table)
