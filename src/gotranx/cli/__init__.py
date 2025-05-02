import typing
from pathlib import Path
import warnings


import typer

from ..schemes import Scheme, get_scheme
from ..codegen import PythonFormat, CFormat
from . import gotran2c, gotran2py, gotran2julia
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
    jax: bool = typer.Option(
        False,
        "--jax",
        help="Use JAX",
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
    scheme: typing.Annotated[
        typing.Optional[typing.List[Scheme]],
        typer.Option(help="Numerical scheme for solving the ODE"),
    ] = None,
    stiff_states: typing.Annotated[
        typing.Optional[typing.List[str]],
        typer.Option("-s", "--stiff-states", help="Stiff states for the hybrid rush larsen scheme"),
    ] = None,
    delta: float = typer.Option(
        1e-8,
        help="Delta value for the rush larsen schemes",
    ),
):
    warnings.warn(
        "convert command is deprecated, use ode2c, ode2py, ode2julia or cellml2ode instead",
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
    scheme: typing.Annotated[
        typing.List[Scheme],
        typer.Option(help="Numerical scheme for solving the ODE"),
    ] = [],
    stiff_states: typing.Annotated[
        typing.List[str],
        typer.Option("-s", "--stiff-states", help="Stiff states for the hybrid rush larsen scheme"),
    ] = [],
    delta: float = typer.Option(
        1e-8,
        help="Delta value for the rush larsen schemes",
    ),
    format: PythonFormat = typer.Option(
        PythonFormat.black,
        "--format",
        "-f",
        help="Formatter for the output code",
    ),
    backend: gotran2py.Backend = typer.Option(
        gotran2py.Backend.numpy,
        "--backend",
        "-b",
        help="Backend for the generated code",
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
    py_config = config_data.get("python", {})
    format = PythonFormat(py_config.get("format", format))
    backend = gotran2py.Backend(py_config.get("backend", backend))

    gotran2py.main(
        fname=fname,
        outname=outname,
        scheme=scheme,
        remove_unused=remove_unused,
        verbose=verbose,
        stiff_states=stiff_states,
        delta=delta,
        format=format,
        backend=backend,
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
        ".h",
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
    scheme: typing.Annotated[
        typing.List[Scheme],
        typer.Option(help="Numerical scheme for solving the ODE"),
    ] = [],
    stiff_states: typing.Annotated[
        typing.List[str],
        typer.Option("-s", "--stiff-states", help="Stiff states for the hybrid rush larsen scheme"),
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
    c_config = config_data.get("c", {})
    to = c_config.get("to", to)
    format = CFormat(c_config.get("format", format))

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
def ode2julia(
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
    scheme: typing.Annotated[
        typing.List[Scheme],
        typer.Option(help="Numerical scheme for solving the ODE"),
    ] = [],
    stiff_states: typing.Annotated[
        typing.List[str],
        typer.Option("-s", "--stiff-states", help="Stiff states for the hybrid rush larsen scheme"),
    ] = [],
    delta: float = typer.Option(
        1e-8,
        help="Delta value for the rush larsen schemes",
    ),
    type_stable: bool = typer.Option(
        False,
        "--type-stable",
        help="Add TYPE to the function signature",
    ),
    # format: CFormat = typer.Option(
    #     CFormat.clang_format,
    #     "--format",
    #     "-f",
    #     help="Formatter for the output code",
    # ),
):
    if fname is None:
        return typer.echo("No file specified")

    config_data = utils.read_config(config)
    verbose = config_data.get("verbose", verbose)
    delta = config_data.get("delta", delta)
    stiff_states = config_data.get("stiff_states", stiff_states)
    scheme = config_data.get("scheme", scheme)
    scheme = utils.validate_scheme(scheme)
    # c_config = config_data.get("c", {})
    # format = CFormat(c_config.get("format", format))

    gotran2julia.main(
        fname=fname,
        outname=outname,
        scheme=scheme,
        remove_unused=remove_unused,
        verbose=verbose,
        stiff_states=stiff_states,
        delta=delta,
        type_stable=type_stable,
    )


@app.command()
def list_schemes():
    from rich.console import Console
    from rich.table import Table

    table = Table(title="Scheme")

    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Key", style="magenta")
    table.add_column("Extra args", style="green")

    def get_extra_arg_names(f):
        import inspect

        args = inspect.signature(f).parameters.keys()
        return [a for a in args if a not in ["ode", "dt", "name", "printer", "remove_unused"]]

    for scheme in Scheme:
        if scheme in [Scheme.forward_explicit_euler, Scheme.forward_generalized_rush_larsen]:
            # These are deprecated
            continue
        f = get_scheme(scheme.value)
        extra_args = str(get_extra_arg_names(f))
        table.add_row(
            " ".join(map(str.capitalize, scheme.name.split("_"))), scheme.value, extra_args
        )

    console = Console()
    console.print(table)


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
