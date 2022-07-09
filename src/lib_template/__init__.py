from importlib.metadata import metadata

from . import cli


meta = metadata("lib_template")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]

__all__ = ["cli"]
