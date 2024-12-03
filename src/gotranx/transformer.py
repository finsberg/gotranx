from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple
from typing import TypeVar

import lark

from . import atoms
from . import exceptions
from . import ode_component
from . import units

T = TypeVar("T", atoms.State, atoms.Parameter)


class LarkODE(NamedTuple):
    components: tuple[ode_component.Component, ...]
    comments: tuple[atoms.Comment, ...]


def remove_quotes(s: str) -> str:
    """Remove quotes from a string

    Parameters
    ----------
    s : str
        The string

    Returns
    -------
    str
        The string without quotes
    """
    return s.replace("'", "").replace('"', "")


def get_unit_and_comment_from_assignment(
    s: lark.Tree,
) -> tuple[str | None, atoms.Comment | None]:
    """Get the unit and comment from an assignment

    Parameters
    ----------
    s : lark.Tree
        The tree

    Returns
    -------
    tuple[str | None, atoms.Comment | None]
        The unit and comment
    """
    # If there are less than 3 children, there is no unit
    if len(s.children) >= 3:
        # The third child is the potential unit
        potential_unit = s.children[2]
        # If it's a comment, it's not a unit
        if isinstance(potential_unit, atoms.Comment):
            try:
                # Try to parse the unit
                unit = units.ureg(potential_unit.text)
            except (units.pint.UndefinedUnitError, AttributeError):
                # Not a proper unit so it's a comment
                return None, atoms.Comment(potential_unit.text)
            else:
                if isinstance(unit, units.pint.Quantity):
                    # It's a proper unit
                    return potential_unit.text, None

                # It is something else, so it's a comment
                return None, atoms.Comment(potential_unit.text)

        else:
            return None, None

    return None, None


def find_assignments(
    s: lark.Tree | lark.lexer.Token | None,
    components: tuple[str, ...],
) -> list[atoms.Assignment]:
    """Find assignments in a tree

    Parameters
    ----------
    s : lark.Tree | lark.lexer.Token | None
        The tree
    components : tuple[str, ...]
        List of components

    Returns
    -------
    list[atoms.Assignment]
        List of assignments
    """
    if isinstance(s, lark.Tree):
        unit_str, comment = get_unit_and_comment_from_assignment(s)
        return [
            atoms.Assignment(
                name=str(s.children[0]),
                value=atoms.Expression(tree=s.children[1]),
                components=tuple(components),
                unit_str=unit_str,
                comment=comment,
            ),
        ]

    return []


def find_components(
    s: list[None | lark.tree.Tree | lark.lexer.Token],
) -> tuple[int, tuple[str, ...]]:
    """Find components in a list

    Parameters
    ----------
    s : list[None  |  lark.tree.Tree  |  lark.lexer.Token]
        The list

    Returns
    -------
    tuple[int, tuple[str, ...]]
        The index and the components
    """
    i = 0
    components = []

    while (
        isinstance(s[i], lark.lexer.Token)
        and hasattr(s[i], "type")
        and s[i].type == "COMPONENT_NAME"  # type:ignore
    ):
        components.append(remove_quotes(str(s[i])))
        i += 1

    # If no components are found, add an empty string
    if len(components) == 0:
        components.append("")
    return i, tuple(components)


def lark_list_to_parameters(
    s: list[None | lark.tree.Tree | lark.lexer.Token],
    cls: type[T],
) -> tuple[T, ...]:
    """Convert a list of lark trees to parameters

    Parameters
    ----------
    s : list[None  |  lark.tree.Tree  |  lark.lexer.Token]
        The list
    cls : type[T]
        The class of the parameters

    Returns
    -------
    tuple[T, ...]
        The parameters
    """
    i, components = find_components(s)
    return tuple(
        [
            tree2parameter(p, components=components, cls=cls)
            for p in s[i:]
            if isinstance(p, lark.Tree)
        ],
    )


def tree2parameter(
    s: lark.Tree,
    components: tuple[str, ...],
    cls: type[T],
) -> T:
    """Convert a lark tree to a parameter

    Parameters
    ----------
    s : lark.Tree
        The tree
    components : tuple[str, ...]
        The components
    cls : type[T]
        The class of the parameter

    Returns
    -------
    T
        The parameter

    Raises
    ------
    exceptions.UnknownTreeTypeError
        If the tree type is unknown
    """
    from .expressions import build_expression

    if s.data == "param":
        return cls(
            name=str(s.children[0]),
            value=build_expression(s.children[1]),
            components=tuple(components),
        )
    elif s.data == "scalarparam":
        unit = None
        if s.children[2] is not None:
            unit = remove_quotes(str(s.children[2]))

        desc = None
        if s.children[3] is not None:
            desc = remove_quotes(str(s.children[3]))

        return cls(
            name=str(s.children[0]),
            value=build_expression(s.children[1]),
            components=tuple(components),
            unit_str=unit,
            description=desc,
        )
    else:
        raise exceptions.UnknownTreeTypeError(datatype=s.data, atom="Parameter")


class TreeToODE(lark.Transformer):
    """Transform a lark tree to an ODE

    See https://lark-parser.readthedocs.io/en/latest/recipes.html
    for more information on how to use lark transformers.
    """

    def _call_userfunc(self, tree, new_children=None):  # pragma: nocover
        # Assumes tree is already transformed
        children = new_children if new_children is not None else tree.children
        try:
            f = getattr(self, tree.data)
        except AttributeError:
            return self.__default__(tree.data, children, tree.meta)
        else:
            try:
                wrapper = getattr(f, "visit_wrapper", None)
                if wrapper is not None:
                    return f.visit_wrapper(f, tree.data, children, tree.meta)
                else:
                    return f(children)
            except lark.GrammarError:
                raise

    def _call_userfunc_token(self, token):  # pragma: nocover
        try:
            f = getattr(self, token.type)
        except AttributeError:
            return self.__default_token__(token)
        else:
            try:
                return f(token)
            except lark.GrammarError:
                raise

    def comment(self, s):
        """Convert a comment to an atoms.Comment"""
        return atoms.Comment(" ".join(map(str.lstrip, map(lambda x: x.lstrip("#"), map(str, s)))))

    def states(self, s: list[None | lark.tree.Tree | lark.lexer.Token]) -> tuple[atoms.State, ...]:
        """Convert a list of states to atoms.State"""
        return lark_list_to_parameters(s, cls=atoms.State)

    def parameters(
        self, s: list[None | lark.tree.Tree | lark.lexer.Token]
    ) -> tuple[atoms.Parameter, ...]:
        """Convert a list of parameters to atoms.Parameter"""
        return lark_list_to_parameters(s, cls=atoms.Parameter)

    def expressions(self, s) -> tuple[atoms.Assignment, ...]:
        """Convert a list of expressions to atoms.Assignment"""
        i, components = find_components(s)

        assignments = []

        for si in s[i:]:
            assignments.extend(find_assignments(si, components=components))

        return tuple(assignments)

    def ode(self, s) -> LarkODE:
        """Convert a lark tree to an ODE

        Parameters
        ----------
        s : list[Any]
            List of objects that could by states, parameters or assignments

        Returns
        -------
        LarkODE
            A tuple of components and comments
        """
        # FIXME: Could use Enum here
        mapping = {
            atoms.Parameter: "parameters",
            atoms.Assignment: "assignments",
            atoms.State: "states",
        }

        components: dict[str, dict[str, set[atoms.Atom]]] = defaultdict(
            lambda: {atom: set() for atom in mapping.values()},
        )

        # breakpoint()

        comments = []
        for line in s:  # Each line in the block
            if isinstance(line, atoms.Comment):
                comments.append(line)
                continue
            if isinstance(line, str):
                assert line.strip() == "", f"Invalid line {line!r}"
                # Skip empty lines
                continue

            for atom in line:  # State, Parameters or Assignment
                for component in atom.components:
                    components[component][mapping[type(atom)]].add(atom)

        # Make sets frozen
        frozen_components: dict[str, dict[str, frozenset[atoms.Atom]]] = {}
        for component_name, component_values in components.items():
            frozen_components[component_name] = {}
            for atom_name, atom_values in component_values.items():
                frozen_components[component_name][atom_name] = frozenset(atom_values)

        # FIXME: Need to somehow tell the type checker that each of the inner dictionaries
        # are actually of the correct type.
        return LarkODE(
            components=tuple(
                [
                    ode_component.Component(name=key, **value)  # type: ignore
                    for key, value in frozen_components.items()
                ],
            ),
            comments=tuple(comments),
        )
