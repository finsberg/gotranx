from __future__ import annotations

import attr
import lark
import pint
import sympy as sp
from structlog import get_logger

from .expressions import build_expression
from .units import ureg
from . import exceptions

logger = get_logger()


def _set_symbol(instance, name: str) -> None:
    """Helper function to set the symbol attribute of a frozen instance

    Parameters
    ----------
    instance : Any
        The instance to set the symbol attribute on
    name : str
        The name of the symbol
    """
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
    """Create a pint unit from a string

    Parameters
    ----------
    unit_str : str | None
        The string representation of the unit

    Returns
    -------
    pint.Unit | None
        It the unit is valid, return the pint unit, else None
    """
    if unit_str is not None:
        try:
            unit = ureg.Unit(unit_str)
        except pint.UndefinedUnitError:
            logger.warning(f"Undefined unit {unit_str!r}")
            unit = None
        except ValueError:
            try:
                unit = ureg.Unit(unit_str.split(" ")[0])
            except Exception:
                logger.warning(f"Invalid unit {unit_str!r}")
                unit = None
    else:
        unit = None
    return unit


def _set_unit(instance, unit_str: str) -> None:
    """Helper function to set the unit attribute of a frozen instance
    by parsing a string representation of the unit

    Parameters
    ----------
    instance : Any
        The instance to set the unit attribute on
    unit_str : str
        The string representation of the unit
    """
    object.__setattr__(instance, "unit", unit_from_string(unit_str))


@attr.s(frozen=True, slots=True)
class Comment:
    """A comment is a string that is not parsed by the parser. It is
    used to add human readable information to the model.
    """

    text: str = attr.ib()


@attr.s(frozen=True, kw_only=True, slots=True)
class Atom:
    """Base class for atoms"""

    name: str = attr.ib()
    value: float | Expression | sp.core.Number | None = attr.ib()
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

    value: Expression | None = attr.ib()
    expr: sp.Expr = attr.ib(sp.S.Zero)
    comment: Comment | None = attr.ib(None)

    def resolve_expression(self, symbols: dict[str, sp.Symbol]) -> Assignment:
        """Resolve the expression of the assignment by
        building the sympy expression from the expression tree

        Parameters
        ----------
        symbols : dict[str, sp.Symbol]
            A dictionary of all symbols in the model

        Returns
        -------
        Assignment
            A new Assignment object with the resolved expression

        Raises
        ------
        exceptions.ResolveExpressionError
            If the expression is not set
        """
        if self.value is None:
            raise exceptions.ResolveExpressionError(name=self.name)
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
            comment=self.comment,
        )

    def to_intermediate(self) -> "Intermediate":
        """Convert the Assignment to an Intermediate"""
        return Intermediate(
            name=self.name,
            value=self.value,
            components=self.components,
            unit_str=self.unit_str,
            unit=self.unit,
            expr=self.expr,
            description=self.description,
            symbol=self.symbol,
            comment=self.comment,
        )

    def to_state_derivative(self, state: State) -> "StateDerivative":
        """Convert the Assignment to a StateDerivative

        Parameters
        ----------
        state : State
            The associated state
        """
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
            comment=self.comment,
        )

    def simplify(self) -> "Assignment":
        """Simplify the expression of the assignment using sympy's
        simplify function. This function returns a new Assignment
        object with the simplified expression.
        """
        return type(self)(
            name=self.name,
            value=self.value,
            components=self.components,
            unit_str=self.unit_str,
            unit=self.unit,
            expr=self.expr.simplify(),
            description=self.description,
            symbol=self.symbol,
            comment=self.comment,
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
        if self.value is None:
            raise exceptions.ResolveExpressionError(name=self.name)
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
            comment=self.comment,
        )
