from dataclasses import dataclass
from pathlib import Path


# Need to inherit from BaseException because
# lark catches all Exceptions in the Transformer
# class
class GotranParserError(BaseException):
    pass


@dataclass
class ODEFileNotFound(GotranParserError):
    fname: Path


@dataclass
class StateNotFound(GotranParserError):
    state_name: str

    def __str__(self) -> str:
        return f"State with name {self.state_name!r} not found"
