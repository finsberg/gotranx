from importlib.metadata import metadata

from . import cli
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
from . import myokit
from .load import load_ode
from .schemes import get_scheme
from .ode import ODE
from .ode_component import Component
from .parser import Parser
from .transformer import TreeToODE


meta = metadata("gotranx")
__version__ = meta["Version"]
__author__ = meta["Author-email"]
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
    "sympytools",
    "schemes",
    "templates",
    "myokit",
    "get_scheme",
]

import structlog as _structlog
import logging as _logging

_structlog.configure(
    wrapper_class=_structlog.make_filtering_bound_logger(_logging.INFO),
)
