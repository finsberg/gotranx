import typing
from pathlib import Path
import warnings

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated  # type: ignore

import typer

from ..schemes import Scheme
from ..codegen import PythonFormat, CFormat
from . import gotran2c, gotran2py
from . import utils

app = typer.Typer()


def version_callback(show_version: bool):
    """Prints version information."""
    if show_version:
        from .. import __version__, __program_name__

        typer.echo(f"{__program_name__} {__version__}")
        raise typer.Exit()


def license_callback(show_license: bool):
    """Prints license information."""
    if show_license:
        from .. import __license__

        typer.echo(f"{__license__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
    license: bool = typer.Option(
        None,
        "--license",
        callback=license_callback,
        is_eager=True,
        help="Show license",
    ),
): ...


@app.command()
def convert(
    fname: typing.Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    to: str = typer.Option(
        "",
        "--to",
        help="Generate code to another programming language",
    ),
    outname: typing.Optional[str] = typer.Option(
        None,
        "-o",
        "--outname",
        help="Output name",
    ),
    remove_unused: bool = typer.Option(
        False,
        "--remove-unused",
        help="Remove unused variables",
    ),
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
    license: bool = typer.Option(
        None,
        "--license",
        callback=license_callback,
        is_eager=True,
        help="Show license",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
    scheme: Annotated[
        typing.Optional[typing.List[Scheme]],
        typer.Option(help="Numerical scheme for solving the ODE"),
    ] = None,
    stiff_states: Annotated[
        typing.Optional[typing.List[str]],
        typer.Option(help="Stiff states for the hybrid rush larsen scheme"),
    ] = None,
    delta: float = typer.Option(
        1e-8,
        help="Delta value for the rush larsen schemes",
    ),
):
    warnings.warn(
        "convert command is deprecated, use ode2c, ode2py or cellml2ode instead",
        DeprecationWarning,
        stacklevel=1,
    )

    if fname is None:
        return typer.echo("No file specified")

    if to == "":
        # Check if outname is specified
        if outname is None:
            return typer.echo("No output name specified")
        else:
            to = Path(outname).suffix

    if to in {".c", ".h", "c"}:
        gotran2c.main(
            fname=fname,
            suffix=to,
            outname=outname,
            scheme=scheme,
            remove_unused=remove_unused,
            verbose=verbose,
            stiff_states=stiff_states,
            delta=delta,
        )
    if to in {".py", "python", "py"}:
        gotran2py.main(
            fname=fname,
            suffix=to,
            outname=outname,
            scheme=scheme,
            remove_unused=remove_unused,
            verbose=verbose,
            stiff_states=stiff_states,
            delta=delta,
        )

    if to in {".ode"}:
        from .cellml2ode import main as _main

        _main(fname=fname, outname=outname, verbose=verbose)


@app.command()
def cellml2ode(
    fname: typing.Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    outname: typing.Optional[str] = typer.Option(
        None,
        "-o",
        "--outname",
        help="Output name",
    ),
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
    license: bool = typer.Option(
        None,
        "--license",
        callback=license_callback,
        is_eager=True,
        help="Show license",
    ),
    config: typing.Optional[Path] = typer.Option(
        None,
        "--config",
        help="Read configuration options from a configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
):
    if fname is None:
        return typer.echo("No file specified")

    config_data = utils.read_config(config)
    verbose = config_data.get("verbose", verbose)

    from .cellml2ode import main as _main

    _main(fname=fname, outname=outname, verbose=verbose)


@app.command()
def ode2py(
    fname: typing.Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    outname: typing.Optional[str] = typer.Option(
        None,
        "-o",
        "--outname",
        help="Output name",
    ),
    remove_unused: bool = typer.Option(
        False,
        "--remove-unused",
        help="Remove unused variables",
    ),
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
    license: bool = typer.Option(
        None,
        "--license",
        callback=license_callback,
        is_eager=True,
        help="Show license",
    ),
    config: typing.Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        help="Read configuration options from a configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
    scheme: Annotated[
        typing.List[Scheme],
        typer.Option(help="Numerical scheme for solving the ODE"),
    ] = [],
    stiff_states: Annotated[
        typing.List[str],
        typer.Option(help="Stiff states for the hybrid rush larsen scheme"),
    ] = [],
    delta: float = typer.Option(
        1e-8,
        help="Delta value for the rush larsen schemes",
    ),
    format: Annotated[
        PythonFormat,
        typer.Option(help="Formatter for the output code"),
    ] = PythonFormat.black,
):
    if fname is None:
        return typer.echo("No file specified")

    config_data = utils.read_config(config)
    verbose = config_data.get("verbose", verbose)
    delta = config_data.get("delta", delta)
    stiff_states = config_data.get("stiff_states", stiff_states)
    scheme = config_data.get("scheme", scheme)
    scheme = utils.validate_scheme(scheme)
    format = PythonFormat[config_data.get("format", {}).get("python", format)]

    gotran2py.main(
        fname=fname,
        outname=outname,
        scheme=scheme,
        remove_unused=remove_unused,
        verbose=verbose,
        stiff_states=stiff_states,
        delta=delta,
        format=format,
    )


@app.command()
def ode2c(
    fname: typing.Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
    to: str = typer.Option(
        "",
        "--to",
        help="Generate code to another programming language",
    ),
    outname: typing.Optional[str] = typer.Option(
        None,
        "-o",
        "--outname",
        help="Output name",
    ),
    remove_unused: bool = typer.Option(
        False,
        "--remove-unused",
        help="Remove unused variables",
    ),
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show version",
    ),
    license: bool = typer.Option(
        None,
        "--license",
        callback=license_callback,
        is_eager=True,
        help="Show license",
    ),
    config: typing.Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        help="Read configuration options from a configuration file",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Verbose output",
    ),
    scheme: Annotated[
        typing.List[Scheme],
        typer.Option(help="Numerical scheme for solving the ODE"),
    ] = [],
    stiff_states: Annotated[
        typing.List[str],
        typer.Option(help="Stiff states for the hybrid rush larsen scheme"),
    ] = [],
    delta: float = typer.Option(
        1e-8,
        help="Delta value for the rush larsen schemes",
    ),
    format: CFormat = typer.Option(
        CFormat.clang_format,
        "--format",
        "-f",
        help="Formatter for the output code",
    ),
):
    if fname is None:
        return typer.echo("No file specified")

    config_data = utils.read_config(config)
    verbose = config_data.get("verbose", verbose)
    delta = config_data.get("delta", delta)
    stiff_states = config_data.get("stiff_states", stiff_states)
    scheme = config_data.get("scheme", scheme)
    scheme = utils.validate_scheme(scheme)
    format = CFormat[config_data.get("format", {}).get("c", format)]

    gotran2c.main(
        fname=fname,
        suffix=to,
        outname=outname,
        scheme=scheme,
        remove_unused=remove_unused,
        verbose=verbose,
        stiff_states=stiff_states,
        delta=delta,
    )


@app.command()
def inspect(
    fname: typing.Optional[Path] = typer.Argument(
        None,
        exists=True,
        file_okay=True,
        dir_okay=False,
        writable=False,
        readable=True,
        resolve_path=True,
    ),
):
    typer.echo("Hello from inspect")
