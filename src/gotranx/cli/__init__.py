from pathlib import Path

import typer


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
    fname: Path | None = typer.Argument(
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
    outname: str | None = typer.Option(
        None,
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
):
    if fname is None:
        return
    if to in [".c", ".h"]:
        from . import gotran2c

        gotran2c.main(fname=fname, suffix=to, outname=outname)
