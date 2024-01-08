from __future__ import annotations

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


def unit_from_string(unit_str: str | None) -> pint.Unit | None:
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
    value: float | Expression | sp.core.Number = attr.ib()
    components: tuple[str, ...] = attr.ib(default=("",))
    description: str | None = attr.ib(None)
    symbol: sp.Symbol = attr.ib(None)
    unit_str: str | None = attr.ib(None, repr=False)
    unit: pint.Unit | None = attr.ib(None)

    def __attrs_post_init__(self):
        if self.unit is None:
            _set_unit(self, unit_str=self.unit_str)
        if self.symbol is None:
            _set_symbol(self, name=self.name)


@attr.s(frozen=True, kw_only=True, slots=True)
class Parameter(Atom):
    """A Parameter is a constant scalar value"""

    value: float | sp.core.Number = attr.ib()


@attr.s(frozen=True, kw_only=True, slots=True)
class State(Atom):
    """A State is a variable that also has a
    corresponding state derivative.
    """

    value: float | sp.core.Number = attr.ib()

    def to_TimeDependentState(self, t: sp.Symbol) -> "TimeDependentState":
        return TimeDependentState(
            name=self.name,
            value=self.value,
            symbol=sp.Function(self.symbol)(t),
            components=self.components,
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

    tree: lark.Tree = attr.ib(cmp=False)  # Different trees can give same expression
    dependencies: frozenset[str] = attr.ib(init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "dependencies", self._find_dependencies())

    def _find_dependencies(self) -> frozenset[str]:
        deps = set()

        for tree in self.tree.iter_subtrees():
            if tree.data == "variable":
                deps.add(str(tree.children[0]))
        return frozenset(deps)

    def resolve(self, symbols: dict[str, sp.Symbol]):
        return build_expression(self.tree, symbols=symbols)


@attr.s(frozen=True, kw_only=True, slots=True)
class Assignment(Atom):
    """Assignments are object of the form `name = value`."""

    value: Expression = attr.ib()
    expr: sp.Expr = attr.ib(sp.S.Zero)

    def resolve_expression(self, symbols: dict[str, sp.Symbol]) -> Assignment:
        expr = self.value.resolve(symbols)
        return type(self)(
            name=self.name,
            value=self.value,
            components=self.components,
            unit_str=self.unit_str,
            unit=self.unit,
            expr=expr,
            symbol=self.symbol,
            description=self.description,
        )

    def to_intermediate(self) -> "Intermediate":
        return Intermediate(
            name=self.name,
            value=self.value,
            components=self.components,
            unit_str=self.unit_str,
            unit=self.unit,
            expr=self.expr,
            description=self.description,
            symbol=self.symbol,
        )

    def to_state_derivative(self, state: State) -> "StateDerivative":
        return StateDerivative(
            name=self.name,
            value=self.value,
            components=self.components,
            unit_str=self.unit_str,
            unit=self.unit,
            state=state,
            expr=self.expr,
            description=self.description,
            symbol=self.symbol,
        )

    def simplify(self) -> "Assignment":
        return type(self)(
            name=self.name,
            value=self.value,
            components=self.components,
            unit_str=self.unit_str,
            unit=self.unit,
            expr=self.expr.simplify(),
            description=self.description,
            symbol=self.symbol,
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

    def resolve_expression(self, symbols: dict[str, sp.Symbol]) -> Assignment:
        expr = self.value.resolve(symbols)
        return StateDerivative(
            name=self.name,
            value=self.value,
            components=self.components,
            unit_str=self.unit_str,
            unit=self.unit,
            symbol=self.symbol,
            expr=expr,
            state=self.state,
            description=self.description,
        )
