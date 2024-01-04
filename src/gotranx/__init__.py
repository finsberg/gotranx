from importlib.metadata import metadata

from . import cli
from . import codecomponent
from . import exceptions
from . import load
from . import ode
from . import ode_component
from . import parser
from . import codegen
from . import transformer
from . import units
from . import sympytools
from . import schemes
from . import templates
from . import cellml
from .load import load_ode
from .ode import ODE
from .ode_component import Component
from .parser import Parser
from .transformer import TreeToODE


meta = metadata("gotranx")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = [
    "cli",
    "parser",
    "Parser",
    "codegen",
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
    "codecomponent",
    "sympytools",
    "schemes",
    "templates",
    "cellml",
]
