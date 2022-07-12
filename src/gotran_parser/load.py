from pathlib import Path

from . import exceptions
from .parser import Parser
from .transformer import TreeToODE


def load_ode(path: str):
    fname = Path(path)

    if not fname.is_file():
        raise exceptions.ODEFileNotFound(fname)

    parser = Parser(parser="lalr", transformer=TreeToODE())
    with open(fname, "r") as f:
        result = parser.parse(f.read())

    return result
