from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple
from typing import Type
from typing import TypeVar

import lark

from . import atoms
from . import exceptions
from . import ode_component

T = TypeVar("T", atoms.State, atoms.Parameter)


class LarkODE(NamedTuple):
    components: tuple[ode_component.Component, ...]
    comments: tuple[atoms.Comment, ...]


def remove_quotes(s: str) -> str:
    return s.replace("'", "").replace('"', "")


def get_unit_from_assignment(s: lark.Tree) -> str | None:
    if len(s.children) >= 3:
        unit = s.children[2]
        try:
            if unit.type == "UNIT":
                return unit.value
        except AttributeError:
            pass
    return None


def find_assignments(
    s,
    components: tuple[str, ...],
) -> list[atoms.Assignment]:
    if isinstance(s, lark.Tree):
        return [
            atoms.Assignment(
                name=str(s.children[0]),
                value=atoms.Expression(tree=s.children[1]),
                components=tuple(components),
                unit_str=get_unit_from_assignment(s),
            ),
        ]

    return []


def find_components(
    s: list[None | lark.tree.Tree | lark.lexer.Token],
) -> tuple[int, tuple[str, ...]]:
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
    cls: Type[T],
) -> tuple[T, ...]:
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
    cls: Type[T],
) -> T:
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
        return atoms.Comment(" ".join(map(str.lstrip, map(lambda x: x.lstrip("#"), map(str, s)))))

    def states(self, s: list[None | lark.tree.Tree | lark.lexer.Token]) -> tuple[atoms.State, ...]:
        return lark_list_to_parameters(s, cls=atoms.State)

    def parameters(
        self, s: list[None | lark.tree.Tree | lark.lexer.Token]
    ) -> tuple[atoms.Parameter, ...]:
        return lark_list_to_parameters(s, cls=atoms.Parameter)

    def expressions(self, s) -> tuple[atoms.Assignment, ...]:
        i, components = find_components(s)

        assignments = []

        for si in s[i:]:
            assignments.extend(find_assignments(si, components=components))

        return tuple(assignments)

    def ode(self, s) -> LarkODE:
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
