import typing

from . import c
from . import py


class Template(typing.Protocol):
    INIT_STATE_VALUES: str
    INIT_PARAMETER_VALUES: str
    RHS: str
    METHOD: str


__all__ = ["c", "py", "Template"]
