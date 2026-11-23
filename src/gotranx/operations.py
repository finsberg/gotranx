"""Operations that build new ODEs from existing ones."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence

import math

import attr
import sympy as sp

from . import atoms
from .atoms import make_symbol
from .codegen.ode import GotranODECodePrinter
from .exceptions import SplitError
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
    is printed in a block naming all of them. The comments are not parsed again:
    the result has ``comments`` as given, so that each is saved as it was (see
    :class:`gotranx.atoms.Comment`).
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
    printer = GotranODECodePrinter(ODE(components, name=name))
    text = "".join(
        [printer.print_states(), printer.print_parameters(), printer.print_assignments()]
    )
    ode = load.ode_from_string(text, name=name, remove_singularities=False)
    return ODE(components=ode.components, t=ode.t, name=name, comments=tuple(comments))


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


def rescale_time(ode: ODE, k: float, components: Iterable[str] | None = None) -> ODE:
    """Return a copy of ``ode`` whose time is rescaled by ``k``.

    With ``t_new = t_old / k``, e.g. ``k = 0.001`` brings a model written in
    seconds onto a clock in milliseconds. Every state derivative in the selected
    components is multiplied by ``k`` and ``t`` is replaced by ``k * t`` in every
    expression of those components. ``components=None`` selects all components.
    """
    if not (math.isfinite(k) and k > 0):
        raise ValueError(f"k must be positive and finite, got {k}")
    selected = None if components is None else set(components)
    time_symbols = [s for name in ("t", "time") if (s := ode.symbols.get(name)) is not None]
    subs = {s: k * s for s in time_symbols}

    new_atoms: list[atoms.Atom] = []
    for atom in all_atoms(ode):
        if not isinstance(atom, atoms.Assignment) or (
            selected is not None and not selected & set(atom.components)
        ):
            new_atoms.append(atom)
            continue
        expr = atom.expr.xreplace(subs)
        if isinstance(atom, atoms.StateDerivative):
            expr = k * expr
        new_atoms.append(attr.evolve(atom, expr=expr))
    return rebuild(new_atoms, name=ode.name, comments=ode.comments)


def _dependencies(atom: atoms.Atom) -> frozenset[str]:
    """The names an assignment reads (none for a state or a parameter)"""
    if not isinstance(atom, atoms.Assignment):
        return frozenset()
    if atom.value is not None:
        return atom.value.dependencies
    # Assignments imported through myokit carry no parse tree.
    return frozenset(str(s) for s in atom.expr.free_symbols)


def _parameter_only(
    lookup: Mapping[str, atoms.Atom], deps: Mapping[str, frozenset[str]]
) -> set[str]:
    """The intermediates that depend, recursively, only on parameters"""
    found: set[str] = set()
    candidates = [n for n, a in lookup.items() if isinstance(a, atoms.Intermediate)]
    changed = True
    while changed:
        changed = False
        for n in candidates:
            if n not in found and all(
                d in found or isinstance(lookup.get(d), atoms.Parameter) for d in deps[n]
            ):
                found.add(n)
                changed = True
    return found


def _monitors(ode: ODE, deps: Mapping[str, frozenset[str]]) -> set[str]:
    """The intermediates that no state derivative depends on, directly or through other
    intermediates"""
    intermediates = {i.name for i in ode.intermediates}
    feeding: set[str] = set()
    todo = [d.name for d in ode.state_derivatives]
    while todo:
        for dep in deps[todo.pop()]:
            if dep in intermediates and dep not in feeding:
                feeding.add(dep)
                todo.append(dep)
    return intermediates - feeding


def split(ode: ODE, name: str, inputs: Iterable[str], outputs: Iterable[str]) -> ODE:
    """Return a copy of ``ode`` with a new component ``name`` carved out of it.

    The new component holds the ``outputs`` and everything they depend on, following
    dependencies until an input or a parameter:

    - A state brings its derivative, and a derivative its state.
    - States, derivatives and intermediates are followed only inside the components the
      outputs belong to. Reaching one of another component is an error: add it to
      ``inputs``.
    - An intermediate that depends, recursively, only on parameters is followed from any
      component. If the rest of the model uses it too, it is computed on both sides
      (written under its old components and ``name``).
    - A parameter used on both sides is written under both. One used only by the new
      component moves there.
    - A monitor, an intermediate that no state derivative depends on, joins the new
      component if it reads a state, a derivative or an intermediate (other than a
      parameter-only one) of the new component, under the same rules.

    Everything else stays where it was. So the declared inputs are exactly what the new
    component needs from the rest, and the outputs the only things of the new component
    the rest may use (an output need not be used).

    Raises
    ------
    gotranx.exceptions.SplitError
        With every message at once: a name that is not in the model, ``name`` already a
        component, a dependency in another component, an input the new component
        computes or does not need, and the rest using something of the new component that
        is not an output.
    """
    inputs = list(dict.fromkeys(inputs))
    outputs = list(dict.fromkeys(outputs))
    lookup = {a.name: a for a in all_atoms(ode)}

    unknown = [x for x in dict.fromkeys([*inputs, *outputs]) if x not in lookup]
    messages = [f"{x} is not a name of the model" for x in unknown]
    if name in {c.name for c in ode.components}:
        messages.append(f"{name} is already a component of the model: choose another name")
    if unknown:
        raise SplitError(messages)

    deps = {n: _dependencies(a) for n, a in lookup.items()}
    parameter_only = _parameter_only(lookup, deps)
    monitors = _monitors(ode, deps)
    partner = {d.name: d.state.name for d in ode.state_derivatives}
    partner.update({state: derivative for derivative, state in partner.items()})
    home = {c for x in outputs for c in lookup[x].components}
    input_set, output_set = set(inputs), set(outputs)

    # The closure: an ordered set, followed breadth first in a fixed order, so that the
    # messages come out in the same order every time.
    closure: dict[str, None] = {}
    todo: deque[str] = deque()
    used_inputs: set[str] = set()
    reached: dict[str, str] = {}

    def add(x: str) -> None:
        if x not in closure:
            closure[x] = None
            todo.append(x)
            if x in partner:
                add(partner[x])

    def follow() -> None:
        while todo:
            for dep in sorted(deps[todo.popleft()]):
                atom = lookup.get(dep)
                if dep in input_set:
                    used_inputs.add(dep)
                elif atom is None or isinstance(atom, atoms.Parameter):
                    continue  # a parameter, time, or a name the model already lacks
                elif dep in parameter_only or home & set(atom.components):
                    add(dep)
                else:
                    where = ", ".join(atom.components)
                    reached.setdefault(
                        dep, f"{name} needs {dep} from component {where}: add it to inputs"
                    )

    def carved(x: str) -> bool:
        """Whether reading ``x`` ties a monitor to the new component"""
        return (
            x in closure and x not in parameter_only and not isinstance(lookup[x], atoms.Parameter)
        )

    for x in outputs:
        add(x)
    follow()
    while joining := [
        m
        for m in sorted(monitors)
        if m not in closure and m not in input_set and any(carved(d) for d in deps[m])
    ]:
        for m in joining:
            add(m)
        follow()

    messages.extend(reached.values())
    messages.extend(
        f"{x} is an input of {name}, but {name} computes it: remove it from inputs"
        for x in inputs
        if x in closure
    )
    rest = [a.name for a in (*ode.intermediates, *ode.state_derivatives) if a.name not in closure]
    leaks = {
        dep
        for x in rest
        for dep in deps[x]
        if dep in closure and dep not in output_set and dep not in parameter_only
    }
    messages.extend(f"the rest uses {x} from {name}: add it to outputs" for x in sorted(leaks))
    messages.extend(
        f"{x} is not needed by {name}" for x in inputs if x not in used_inputs and x not in closure
    )
    if messages:
        raise SplitError(messages)

    # What each side reads. The rest also reads through the parameter-only intermediates
    # it shares with the new component.
    shared = parameter_only - output_set
    rest_uses = {dep for x in rest for dep in deps[x]}
    through = [x for x in rest_uses if x in closure and x in shared]
    while through:
        for dep in deps[through.pop()]:
            if dep not in rest_uses:
                rest_uses.add(dep)
                if dep in closure and dep in shared:
                    through.append(dep)
    new_uses = {dep for x in closure for dep in deps[x]} - input_set

    def both(atom: atoms.Atom) -> tuple[str, ...]:
        return (*atom.components, name)

    def place(atom: atoms.Atom) -> atoms.Atom:
        n = atom.name
        if n in closure:
            components = both(atom) if n in shared and n in rest_uses else (name,)
        elif isinstance(atom, atoms.Parameter) and n in new_uses:
            components = both(atom) if n in rest_uses else (name,)
        else:
            return atom
        if isinstance(atom, atoms.StateDerivative):
            state = attr.evolve(atom.state, components=components)
            return attr.evolve(atom, components=components, state=state)
        return attr.evolve(atom, components=components)

    return rebuild([place(a) for a in all_atoms(ode)], name=ode.name, comments=ode.comments)
