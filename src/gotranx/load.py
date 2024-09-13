from __future__ import annotations
from pathlib import Path

from structlog import get_logger

from . import exceptions
from .ode import make_ode, ODE
from .parser import Parser
from .transformer import TreeToODE, LarkODE


logger = get_logger()


def ode_from_string(text: str, name="ode") -> ODE:
    """Create an ODE from a string

    Parameters
    ----------
    text : str
        The string to parse
    name : str, optional
        Name of the ODE, by default "ode"

    Returns
    -------
    gotranx.ode.ODE
        The ODE
    """
    parser = Parser(parser="lalr", transformer=TreeToODE(), propagate_positions=True)
    result = parser.parse(text)
    if not isinstance(result, LarkODE):
        raise exceptions.InvalidODEException(text=text, atoms=result)

    ode = make_ode(
        components=result.components,
        name=name,
        comments=result.comments,
    )
    logger.info(f"Num states {ode.num_states}")
    logger.info(f"Num parameters {ode.num_parameters}")
    return ode


def load_ode(path: str | Path) -> ODE:
    """Load an ODE from a file

    Parameters
    ----------
    path : str | Path
        Path to the file

    Returns
    -------
    gotranx.ode.ODE
        The ODE

    Raises
    ------
    exceptions.ODEFileNotFound
        Raised if the file is not found
    """
    fname = Path(path)

    logger.info(f"Load ode {path}")

    if not fname.is_file():
        raise exceptions.ODEFileNotFound(fname)

    return ode_from_string(fname.read_text(), name=fname.stem)
