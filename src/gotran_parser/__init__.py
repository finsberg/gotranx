from importlib.metadata import metadata

from . import cli
from . import exceptions
from . import load
from . import ode
from . import ode_component
from . import parser
from . import transformer
from . import units
from .load import load_ode
from .ode import ODE
from .ode_component import Component
from .parser import Parser
from .transformer import TreeToODE


meta = metadata("gotran_parser")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
    "cli",
    "parser",
    "Parser",
    "transformer",
    "TreeToODE",
    "exceptions",
    "load",
    "load_ode",
    "ode_component",
    "Component",
    "ode",
    "ODE",
    "units",
]
