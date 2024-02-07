from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lark


class GotranxError(Exception):
    pass


@dataclass
class ODEFileNotFound(GotranxError):
    fname: Path


@dataclass
class StateNotFoundInComponent(GotranxError):
    state_name: str
    component_name: str

    def __str__(self) -> str:
        return (
            f"State with name {self.state_name!r} "
            f"not found in component {self.component_name!r}"
        )


@dataclass
class ParameterNotFoundInComponent(GotranxError):
    parameter_name: str
    component_name: str

    def __str__(self) -> str:
        return (
            f"Parameter with name {self.parameter_name!r} "
            f"not found in component {self.component_name!r}"
        )


@dataclass
class ComponentNotCompleteError(GotranxError):
    component_name: str
    missing_state_derivatives: list[str]

    def __str__(self) -> str:
        return (
            f"Component {self.component_name!r} is not complete. "
            f"Missing state derivatives for {self.missing_state_derivatives!r}"
        )


@dataclass
class DuplicateSymbolError(GotranxError):
    duplicates: set[str]

    def __str__(self) -> str:
        return (
            f"Found multiple definitions for {self.duplicates!r}. "
            "Please make sure to only define each variable once in order "
            "to avoid ambiguity. Equations should be reorderable."
        )


@dataclass
class UnknownTreeTypeError(GotranxError):
    datatype: str
    atom: str

    def __str__(self) -> str:
        return f"Cannot prase tree data type {self.datatype} for atom {self.atom}"


@dataclass
class InvalidTreeError(GotranxError):
    tree: lark.Tree

    def __str__(self) -> str:
        return f"Invaild tree with data attribute {self.tree.data!r} \n{self.tree.pretty()}"


@dataclass
class MissingSymbolError(GotranxError, KeyError):
    symbol: str
    line_no: int

    def __str__(self) -> str:
        return f"Symbol {self.symbol!r} not found in line {self.line_no}"


@dataclass
class ResolveExpressionError(GotranxError, ValueError):
    name: str

    def __str__(self) -> str:
        return f"Unable to resolve expression for {self.name!r}"
