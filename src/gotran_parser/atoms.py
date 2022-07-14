from __future__ import annotations

from typing import Optional
from typing import Union

import attr
import lark
import pint
import sympy as sp
from structlog import get_logger

from .units import ureg

logger = get_logger()


@attr.s(frozen=True, kw_only=True, slots=True)
class Parameter:
    """A Parameter is a constant scalar value"""

    name: str = attr.ib()
    value: float = attr.ib()
    component: Optional[str] = attr.ib(None)
    description: Optional[str] = attr.ib(None)
    symbol: sp.Symbol = attr.ib(init=False)
    unit_str: Optional[str] = attr.ib(None, repr=False)
    unit: Optional[pint.Unit] = attr.ib(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "unit", unit_from_string(self.unit_str))
        object.__setattr__(
            self,
            "symbol",
            sp.Symbol(
                name=self.name,
                real=True,
                imaginary=False,
                commutative=True,
                finite=True,
            ),
        )


@attr.s(frozen=True, kw_only=True, slots=True)
class State:
    """A State is a variable that also has a
    corresponding state derivative.
    """

    name: str = attr.ib()
    ic: float = attr.ib()
    component: Optional[str] = attr.ib(None)
    info: Optional[str] = attr.ib(None)
    unit: Optional[str] = attr.ib(None)
    description: Optional[str] = attr.ib(None)
    symbol: sp.Symbol = attr.ib(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(
            self,
            "symbol",
            sp.Symbol(
                name=self.name,
                real=True,
                imaginary=False,
                commutative=True,
                finite=True,
            ),
        )


@attr.s(frozen=True, kw_only=True, slots=True)
class Expression:
    """An Expression is a group of variables (i.e
    states, parameters or other expressions) combined
    with binary operations (i.e +, -, * etc)
    An Expression is typically a right hand side of
    an assignment."""

    tree: lark.Tree = attr.ib()
    dependencies: frozenset[str] = attr.ib(init=False)

    def __attrs_post_init__(self):
        self._find_dependencies()

    def _find_dependencies(self):
        deps = set()
        for tree in self.tree.iter_subtrees():
            if tree.data == "name":
                deps.add(str(tree.children[0]))

        object.__setattr__(self, "dependencies", frozenset(deps))


def unit_from_string(unit_str: Optional[str]) -> Optional[pint.Unit]:
    if unit_str is not None:
        try:
            unit = ureg.Unit(unit_str)
        except pint.UndefinedUnitError:
            logger.warning(f"Undefined unit {unit_str!r}")
            unit = None
    else:
        unit = None
    return unit


@attr.s(frozen=True, kw_only=True, slots=True)
class Assignment:
    """Assignments are object of the form `lhs = rhs`."""

    lhs: str = attr.ib()
    rhs: Expression = attr.ib()
    component: Optional[str] = attr.ib(None)
    info: Optional[str] = attr.ib(None)
    unit_str: Optional[str] = attr.ib(None, repr=False)
    unit: Optional[pint.Unit] = attr.ib(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "unit", unit_from_string(self.unit_str))

    def to_intermediate(self) -> "Intermediate":
        return Intermediate(
            lhs=self.lhs,
            rhs=self.rhs,
            component=self.component,
            info=self.info,
            unit=self.unit,
        )

    def to_state_derivative(self, state: State) -> "StateDerivative":
        return StateDerivative(
            lhs=self.lhs,
            rhs=self.rhs,
            component=self.component,
            info=self.info,
            unit=self.unit,
            state=state,
        )


@attr.s(frozen=True, kw_only=True, slots=True)
class Intermediate:
    """Intermediate is a type of Assignment that is not
    a StateDerivative"""

    lhs: str = attr.ib()
    rhs: Expression = attr.ib()
    component: Optional[str] = attr.ib(None)
    info: Optional[str] = attr.ib(None)
    unit: Optional[str] = attr.ib(None)


@attr.s(frozen=True, kw_only=True, slots=True)
class StateDerivative:
    """A StateDerivative is an Assignment of the form
    `dX_dt = rhs` where X is a state. A StatedDerivative
    also holds a pointer to the State"""

    lhs: str = attr.ib()
    rhs: Expression = attr.ib()
    state: State = attr.ib()
    component: Optional[str] = attr.ib(None)
    info: Optional[str] = attr.ib(None)
    unit: Optional[str] = attr.ib(None)


Atoms = Union[State, Parameter, Assignment]
