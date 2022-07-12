from __future__ import annotations

from collections import defaultdict
from typing import Optional

import lark

from . import atoms


def remove_quotes(s: str) -> str:
    return s.replace("'", "").replace('"', "")


def find_assignment_component(s) -> Optional[str]:
    component = None
    if isinstance(s, lark.Token) and s.type == "COMPONENT_NAME":
        component = remove_quotes(str(s))
    return component


def find_assignments(s, component=None):
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
    def states(self, s) -> list[atoms.State]:
        return [
            atoms.State(name=str(p[0]), ic=float(p[1]), component=s[0], info=s[1])
            for p in s[2:]
        ]

    def parameters(self, s) -> list[atoms.Parameter]:
        return [
            atoms.Parameter(name=str(p[0]), value=float(p[1]), component=s[0])
            for p in s[1:]
        ]

    def pair(self, s):
        name, value = s
        return (name, value)

    def expressions(self, s) -> list[atoms.Assignment]:
        component = find_assignment_component(s[0])
        assignments = []

        for si in s:
            assignments.extend(find_assignments(si, component=component))

        return assignments

    def ode(self, s):
        mapping = {
            atoms.Parameter: "parameters",
            atoms.Assignment: "assignments",
            atoms.State: "states",
        }
        components = defaultdict(
            lambda: {"assignments": set(), "parameters": set(), "states": set()},
        )

        for line in s:
            for value in line:
                components[value.component][mapping[type(value)]].add(value)

        return [atoms.Component(name=key, **value) for key, value in components.items()]
