import typing
from pathlib import Path

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated  # type: ignore

import typer
from ..schemes import Scheme

from . import gotran2c, gotran2py, cellml2ode

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


@app.command()
def main(
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
    scheme: Annotated[typing.Optional[typing.List[Scheme]], typer.Option()] = None,
):
    if fname is None:
        return typer.echo("No file specified")

    if to == "":
        # Check if outname is specified
        if outname is None:
            return typer.echo("No output name specified")
        else:
            to = Path(outname).suffix

    if to in {".c", ".h"}:
        gotran2c.main(
            fname=fname,
            suffix=to,
            outname=outname,
            scheme=scheme,
            remove_unused=remove_unused,
            verbose=verbose,
        )
    if to in {".py"}:
        gotran2py.main(
            fname=fname,
            suffix=to,
            outname=outname,
            scheme=scheme,
            remove_unused=remove_unused,
            verbose=verbose,
        )

    if to in {".ode"}:
        cellml2ode.main(fname=fname, outname=outname, verbose=verbose)
