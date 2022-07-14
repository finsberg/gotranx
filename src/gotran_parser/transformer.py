from __future__ import annotations

from collections import defaultdict
from typing import Optional

import lark

from . import atoms
from . import ode


def remove_quotes(s: str) -> str:
    return s.replace("'", "").replace('"', "")


def find_assignment_component(s) -> Optional[str]:
    component = None
    if isinstance(s, lark.Token) and s.type == "COMPONENT_NAME":
        component = remove_quotes(str(s))
    return component


def find_assignments(s, component: Optional[str] = None) -> list[atoms.Assignment]:
    if isinstance(s, lark.Tree):
        return [
            atoms.Assignment(
                lhs=str(s.children[0]),
                rhs=atoms.Expression(tree=s.children[1]),
                component=component,
            ),
        ]

    return []


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

    def states(self, s) -> tuple[atoms.State, ...]:
        component = s[0]
        if component is not None:
            component = remove_quotes(str(component))

        info = s[1]
        if info is not None:
            info = remove_quotes(str(info))
        return tuple(
            [
                atoms.State(
                    name=str(p[0]),
                    ic=float(p[1]),
                    component=component,
                    info=info,
                )
                for p in s[2:]
            ],
        )

    def parameters(self, s) -> tuple[atoms.Parameter, ...]:
        component = s[0]
        if component is not None:
            component = remove_quotes(str(component))
        return tuple(
            [
                atoms.Parameter(name=str(p[0]), value=float(p[1]), component=component)
                for p in s[1:]
            ],
        )

    def pair(self, s):
        return s

    def expressions(self, s) -> tuple[atoms.Assignment, ...]:
        component = find_assignment_component(s[0])
        assignments = []

        for si in s:
            assignments.extend(find_assignments(si, component=component))

        return tuple(assignments)

    def ode(self, s) -> tuple[ode.Component, ...]:

        # FIXME: Could use Enum here
        mapping = {
            atoms.Parameter: "parameters",
            atoms.Assignment: "assignments",
            atoms.State: "states",
        }

        components: dict[Optional[str], dict[str, set[atoms.Atoms]]] = defaultdict(
            lambda: {atom: set() for atom in mapping.values()},
        )

        for line in s:  # Each line in the block
            for atom in line:  # State, Parameters or Assignment
                components[atom.component][mapping[type(atom)]].add(atom)

        # Make sets frozen
        frozen_components: dict[Optional[str], dict[str, frozenset[atoms.Atoms]]] = {}
        for component_name, component_values in components.items():
            frozen_components[component_name] = {}
            for atom_name, atom_values in component_values.items():
                frozen_components[component_name][atom_name] = frozenset(atom_values)

        # FIXME: Need to somehow tell the type checker that each of the inner dictionaries
        # are actually of the correct type.
        return tuple(
            [
                ode.Component(name=key, **value)  # type: ignore
                for key, value in frozen_components.items()
            ],
        )
