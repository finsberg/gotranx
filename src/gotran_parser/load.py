from pathlib import Path

from structlog import get_logger

from . import exceptions
from .ode import make_ode
from .parser import Parser
from .transformer import TreeToODE


logger = get_logger()


def load_ode(path: str):
    fname = Path(path)

    logger.info(f"Load ode {path}")

    if not fname.is_file():
        raise exceptions.ODEFileNotFound(fname)

    parser = Parser(parser="lalr", transformer=TreeToODE())
    with open(fname, "r") as f:
        result = parser.parse(f.read())

    ode = make_ode(
        components=result.components,
        name=fname.stem,
        comments=result.comments,
    )
    logger.info(f"Num states {ode.num_states}")
    logger.info(f"Num parameters {ode.num_parameters}")
    return ode
