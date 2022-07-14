from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Need to inherit from BaseException because
# lark catches all Exceptions in the Transformer
# class. FIXME: Find a way to avoid this
# We would like that users the catch "Exception"
# also catches these exceptions
class GotranParserError(Exception):
    pass


@dataclass
class ODEFileNotFound(GotranParserError):
    fname: Path


@dataclass
class StateNotFoundInComponent(GotranParserError):
    state_name: str
    component_name: str

    def __str__(self) -> str:
        return f"State with name {self.state_name!r} not found in component {self.component_name!r}"


@dataclass
class ComponentNotCompleteError(GotranParserError):
    component_name: str
    missing_state_derivatives: list[str]

    def __str__(self) -> str:
        return (
            f"Component {self.component_name!r} is not complete. "
            f"Missing state derivatives for {self.missing_state_derivatives!r}"
        )


@dataclass
class DuplicateSymbolError(GotranParserError):
    duplicates: list[str]

    def __str__(self) -> str:
        return (
            f"Found multiple definitions for {self.duplicates!r}. "
            "Please make sure to only define each variable once in order "
            "to avoid ambiguity. Equations should be reorderable."
        )
