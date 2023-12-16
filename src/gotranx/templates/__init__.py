from __future__ import annotations
import typing

from . import c
from . import python


class Template(typing.Protocol):
    @staticmethod
    def state_index(data: dict[str, float]) -> str:
        ...

    @staticmethod
    def parameter_index(data: dict[str, float]) -> str:
        ...

    @staticmethod
    def init_state_values(
        name: str, state_values: list[float], state_names: list[str], code: str
    ) -> str:
        ...

    @staticmethod
    def init_parameter_values(
        name: str, parameter_values: list[float], parameter_names: list[str], code: str
    ) -> str:
        ...

    @staticmethod
    def method(
        name: str,
        args: str,
        states: str,
        parameters: str,
        values: str,
        return_name: str | None,
        num_return_values: int,
    ) -> str:
        ...


__all__ = ["c", "python", "Template"]
