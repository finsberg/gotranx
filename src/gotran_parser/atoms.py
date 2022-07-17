from __future__ import annotations

from typing import Optional
from typing import Union

import attr
import lark
import pint
import sympy as sp
from structlog import get_logger

from .expressions import build_expression
from .units import ureg

logger = get_logger()


def _set_symbol(instance, name: str) -> None:
    object.__setattr__(
        instance,
        "symbol",
        sp.Symbol(
            name=name,
            real=True,
            imaginary=False,
            commutative=True,
            finite=True,
        ),
    )


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


def _set_unit(instance, unit_str: str) -> None:
    object.__setattr__(instance, "unit", unit_from_string(unit_str))


@attr.s(frozen=True, slots=True)
class Comment:
    text: str = attr.ib()


@attr.s(frozen=True, kw_only=True, slots=True)
class Atom:
    """Base class for atoms"""

    name: str = attr.ib()
    value: Union[float, Expression] = attr.ib()
    component: Optional[str] = attr.ib(None)
    description: Optional[str] = attr.ib(None)
    info: Optional[str] = attr.ib(None)
    symbol: sp.Symbol = attr.ib(None)
    unit_str: Optional[str] = attr.ib(None, repr=False)
    unit: Optional[pint.Unit] = attr.ib(None)

    def __attrs_post_init__(self):
        if self.unit is None:
            _set_unit(self, unit_str=self.unit_str)
        if self.symbol is None:
            _set_symbol(self, name=self.name)


@attr.s(frozen=True, kw_only=True, slots=True)
class Parameter(Atom):
    """A Parameter is a constant scalar value"""

    value: float = attr.ib()


@attr.s(frozen=True, kw_only=True, slots=True)
class State(Atom):
    """A State is a variable that also has a
    corresponding state derivative.
    """

    value: float = attr.ib()

    def to_TimeDependentState(self, t: sp.Symbol) -> "TimeDependentState":
        return TimeDependentState(
            name=self.name,
            value=self.value,
            symbol=sp.Function(self.symbol)(t),
            component=self.component,
            info=self.info,
            description=self.description,
            unit_str=self.unit_str,
            unit=self.unit,
        )


@attr.s(frozen=True, kw_only=True, slots=True)
class TimeDependentState(State):
    """A TimeDependentState is a State, where the symbol
    is a sympy Function instead of a pure Symbol.
    """

    symbol: sp.Function = attr.ib()


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
        object.__setattr__(self, "dependencies", self._find_dependencies())

    def _find_dependencies(self) -> frozenset[str]:
        deps = set()
        for tree in self.tree.iter_subtrees():
            if tree.data == "name":
                deps.add(str(tree.children[0]))
        return frozenset(deps)

    def resolve(self, symbols: dict[str, sp.Symbol]):
        return build_expression(self.tree, symbols=symbols)


@attr.s(frozen=True, kw_only=True, slots=True)
class Assignment(Atom):
    """Assignments are object of the form `name = value`."""

    value: Expression = attr.ib()
    expr: Optional[sp.Expr] = attr.ib(None)

    def is_resolved(self) -> bool:
        return self.expr is not None

    def resolve_expression(self, symbols: dict[str, sp.Symbol]) -> Assignment:
        expr = self.value.resolve(symbols)
        return Assignment(
            name=self.name,
            value=self.value,
            component=self.component,
            info=self.info,
            unit_str=self.unit_str,
            unit=self.unit,
            expr=expr,
        )

    def to_intermediate(self) -> "Intermediate":
        return Intermediate(
            name=self.name,
            value=self.value,
            component=self.component,
            info=self.info,
            unit_str=self.unit_str,
            unit=self.unit,
            expr=self.expr,
        )

    def to_state_derivative(self, state: State) -> "StateDerivative":
        return StateDerivative(
            name=self.name,
            value=self.value,
            component=self.component,
            info=self.info,
            unit_str=self.unit_str,
            unit=self.unit,
            state=state,
            expr=self.expr,
        )


@attr.s(frozen=True, kw_only=True, slots=True)
class Intermediate(Assignment):
    """Intermediate is a type of Assignment that is not
    a StateDerivative"""


@attr.s(frozen=True, kw_only=True, slots=True)
class StateDerivative(Assignment):
    """A StateDerivative is an Assignment of the form
    `dX_dt = value` where X is a state. A StatedDerivative
    also holds a pointer to the State"""

    state: State = attr.ib()


# @attr.s(frozen=True, kw_only=True, slots=True)
# class TimeDependentStateDerivative(StateDerivative):
#     """A StateDerivative is an Assignment of the form
#     `dX_dt = value` where X is a state. A StatedDerivative
#     also holds a pointer to the State"""

#     state: TimeDependentState = attr.ib()
#     symbol: sp.Symbol = attr.ib(init=False)

#     def __attrs_post_init__(self):
#         object.__setattr__(
#             self, "symbol", sp.Derivative(self.state.symbol)


Atoms = Union[State, Parameter, Assignment, StateDerivative]
