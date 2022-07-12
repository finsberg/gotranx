from importlib.metadata import metadata

from . import cli
from . import exceptions
from . import parser
from . import transformer
from .parser import Parser
from .transformer import TreeToODE


meta = metadata("gotran_parser")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = ["cli", "parser", "Parser", "transformer", "TreeToODE", "exceptions"]
