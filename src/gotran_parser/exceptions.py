from dataclasses import dataclass
from pathlib import Path


class GotranParserError(Exception):
    pass


@dataclass
class ODEFileNotFound(GotranParserError):
    fname: Path
