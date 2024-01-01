from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lark


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
        return (
            f"State with name {self.state_name!r} "
            f"not found in component {self.component_name!r}"
        )


@dataclass
class ParameterNotFoundInComponent(GotranParserError):
    parameter_name: str
    component_name: str

    def __str__(self) -> str:
        return (
            f"Parameter with name {self.parameter_name!r} "
            f"not found in component {self.component_name!r}"
        )


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
    duplicates: set[str]

    def __str__(self) -> str:
        return (
            f"Found multiple definitions for {self.duplicates!r}. "
            "Please make sure to only define each variable once in order "
            "to avoid ambiguity. Equations should be reorderable."
        )


@dataclass
class UnknownTreeTypeError(GotranParserError):
    datatype: str
    atom: str

    def __str__(self) -> str:
        return f"Cannot prase tree data type {self.datatype} for atom {self.atom}"


@dataclass
class InvalidTreeError(GotranParserError):
    tree: lark.Tree

    def __str__(self) -> str:
        return f"Invaild tree with data attribute {self.tree.data!r} \n{self.tree.pretty()}"
