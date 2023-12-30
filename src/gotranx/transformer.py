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


def find_assignment_component(s) -> str | None:
    component = None
    if isinstance(s, lark.Token) and s.type == "COMPONENT_NAME":
        component = remove_quotes(str(s))
    return component


def find_assignment_info(s) -> str | None:
    info = None
    if len(s) > 1 and isinstance(s[1], lark.Token) and s[1].type == "INFO":
        info = remove_quotes(str(s[1]))
    return info


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
    component: str | None = None,
    info: str | None = None,
) -> list[atoms.Assignment]:
    if isinstance(s, lark.Tree):
        return [
            atoms.Assignment(
                name=str(s.children[0]),
                value=atoms.Expression(tree=s.children[1]),
                component=component,
                info=info,
                unit_str=get_unit_from_assignment(s),
            ),
        ]

    return []


def lark_list_to_parameters(
    s: list[None | lark.tree.Tree | lark.lexer.Token],
    cls: Type[T],
) -> tuple[T, ...]:
    component = s[0]
    if component is not None:
        component = remove_quotes(str(component))

    info = s[1]
    if info is not None:
        info = remove_quotes(str(info))

    return tuple(
        [tree2parameter(p, component=component, info=info, cls=cls) for p in s[2:]],
    )


def tree2parameter(
    s: lark.Tree,
    component: str | None,
    cls: Type[T],
    info: str | None = None,
) -> T:
    kwargs = {}
    if info is not None:
        kwargs["info"] = info
    if s.data == "param":
        return cls(
            name=str(s.children[0]),
            value=float(s.children[1]),
            component=component,
            **kwargs,
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
            value=float(s.children[1]),
            component=component,
            unit_str=unit,
            description=desc,
            **kwargs,
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
        return atoms.Comment(" ".join(map(str, s)))

    def states(self, s: list[None | lark.tree.Tree | lark.lexer.Token]) -> tuple[atoms.State, ...]:
        return lark_list_to_parameters(s, cls=atoms.State)

    def parameters(
        self, s: list[None | lark.tree.Tree | lark.lexer.Token]
    ) -> tuple[atoms.Parameter, ...]:
        return lark_list_to_parameters(s, cls=atoms.Parameter)

    def expressions(self, s) -> tuple[atoms.Assignment, ...]:
        component = find_assignment_component(s[0])
        info = find_assignment_info(s)
        assignments = []

        for si in s:
            assignments.extend(find_assignments(si, component=component, info=info))

        return tuple(assignments)

    def ode(self, s) -> LarkODE:
        # FIXME: Could use Enum here
        mapping = {
            atoms.Parameter: "parameters",
            atoms.Assignment: "assignments",
            atoms.State: "states",
        }

        components: dict[str | None, dict[str, set[atoms.Atom]]] = defaultdict(
            lambda: {atom: set() for atom in mapping.values()},
        )
        comments = []
        for line in s:  # Each line in the block
            if isinstance(line, atoms.Comment):
                comments.append(line)
                continue

            for atom in line:  # State, Parameters or Assignment
                components[atom.component][mapping[type(atom)]].add(atom)

        # Make sets frozen
        frozen_components: dict[str | None, dict[str, frozenset[atoms.Atom]]] = {}
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
