"""Operations that build new ODEs from existing ones."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import attr
import sympy as sp

from . import atoms
from .atoms import make_symbol
from .codegen.ode import GotranODECodePrinter
from .ode import ODE
from .ode_component import Component


def all_atoms(ode: ODE) -> tuple[atoms.Atom, ...]:
    """The states, parameters, intermediates and state derivatives of an ODE."""
    return (*ode.states, *ode.parameters, *ode.intermediates, *ode.state_derivatives)


def rebuild(
    atoms_: Iterable[atoms.Atom],
    *,
    name: str,
    comments: Sequence[atoms.Comment] = (),
) -> ODE:
    """Build an ODE from atoms by printing them to ``.ode`` text and parsing it.

    Printing uses ``atom.expr`` and ``atom.value``, so the result is consistent
    even if the atoms were modified after parsing. An atom in several components
    is printed in a block naming all of them.
    """
    from . import load  # lazy: avoids an import cycle

    states: dict[str, set[atoms.State]] = {}
    parameters: dict[str, set[atoms.Parameter]] = {}
    assignments: dict[str, set[atoms.Assignment]] = {}
    for atom in atoms_:
        if isinstance(atom, atoms.State):
            target: dict = states
        elif isinstance(atom, atoms.Parameter):
            target = parameters
        elif isinstance(atom, atoms.Assignment):
            target = assignments
        else:
            raise TypeError(f"Cannot rebuild from {atom!r}")
        for comp in atom.components:
            target.setdefault(comp, set()).add(atom)

    comp_names = list(dict.fromkeys([*states, *parameters, *assignments]))
    components = [
        Component(
            name=c,
            states=frozenset(states.get(c, ())),
            parameters=frozenset(parameters.get(c, ())),
            assignments=frozenset(assignments.get(c, ())),
        )
        for c in comp_names
    ]
    printer = GotranODECodePrinter(ODE(components, name=name, comments=tuple(comments)))
    text = "".join(
        [
            printer.print_comments(),
            printer.print_states(),
            printer.print_parameters(),
            printer.print_assignments(),
        ]
    )
    return load.ode_from_string(text, name=name, remove_singularities=False)


def rename(ode: ODE, mapping: Mapping[str, str]) -> ODE:
    """Return a copy of ``ode`` with atoms renamed according to ``mapping``.

    Renaming a state also renames its derivative. Raises ``KeyError`` for an
    unknown name and ``DuplicateSymbolError`` if a name is then defined twice.
    """
    known = {a.name for a in all_atoms(ode)}
    for old in mapping:
        if old not in known:
            raise KeyError(f"Cannot rename {old!r}: no such name in {ode.name!r}")

    full = dict(mapping)
    for old, new in mapping.items():
        if isinstance(ode[old], atoms.State):
            full.setdefault(f"d{old}_dt", f"d{new}_dt")

    subs = {make_symbol(o): make_symbol(n) for o, n in full.items()}

    def fix_value(value):
        return value.xreplace(subs) if isinstance(value, sp.Basic) else value

    def new_state(s: atoms.State) -> atoms.State:
        n = full.get(s.name, s.name)
        return attr.evolve(s, name=n, symbol=make_symbol(n), value=fix_value(s.value))

    new_atoms: list[atoms.Atom] = []
    for atom in all_atoms(ode):
        n = full.get(atom.name, atom.name)
        if isinstance(atom, atoms.StateDerivative):
            new_atoms.append(
                attr.evolve(
                    atom,
                    name=n,
                    symbol=make_symbol(n),
                    expr=atom.expr.xreplace(subs),
                    state=new_state(atom.state),
                )
            )
        elif isinstance(atom, atoms.Assignment):
            new_atoms.append(
                attr.evolve(atom, name=n, symbol=make_symbol(n), expr=atom.expr.xreplace(subs))
            )
        elif isinstance(atom, atoms.State):
            new_atoms.append(new_state(atom))
        else:
            new_atoms.append(
                attr.evolve(atom, name=n, symbol=make_symbol(n), value=fix_value(atom.value))
            )
    return rebuild(new_atoms, name=ode.name, comments=ode.comments)
