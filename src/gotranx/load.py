from __future__ import annotations
from pathlib import Path

from structlog import get_logger

from . import exceptions
from .ode import make_ode, ODE
from .parser import Parser
from .transformer import TreeToODE


logger = get_logger()


def ode_from_string(text: str, name="ode") -> ODE:
    parser = Parser(parser="lalr", transformer=TreeToODE())
    result = parser.parse(text)
    ode = make_ode(
        components=result.components,
        name=name,
        comments=result.comments,
    )
    logger.info(f"Num states {ode.num_states}")
    logger.info(f"Num parameters {ode.num_parameters}")
    return ode


def load_ode(path: str | Path):
    fname = Path(path)

    logger.info(f"Load ode {path}")

    if not fname.is_file():
        raise exceptions.ODEFileNotFound(fname)

    return ode_from_string(fname.read_text(), name=fname.stem)
