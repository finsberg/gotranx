"""Compose an ODE from named parts and a glue file.

A composition has two files:

- a **recipe** (TOML), which says which models (parts) are combined and how each is
  adapted to the others: its name, its file, its time unit and any renames;
- a **glue file**, which says how they are coupled, in the ``.ode`` grammar plus
  ``part.name``, ``replace`` and ``+=`` (see :mod:`gotranx.glue`).

The result is an ordinary :class:`gotranx.ode.ODE`, in which each part is exactly one
component, named after the part. See :func:`compose`.
"""

from __future__ import annotations

import graphlib
import hashlib
import json
import math
import tomllib
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import attr
import lark
import sympy as sp

from . import atoms
from . import exceptions
from .atoms import make_symbol
from .expressions import build_expression
from .glue import GlueStatement, load_glue, unqualify
from .load import load_ode
from .ode import ODE
from .ode_component import STATE_DERIV_EXPR
from .operations import (
    TIME_NAMES,
    VALID_NAME,
    _listing,
    all_atoms,
    rebuild,
    rename,
    rescale_time,
)
from .units import ureg

RECIPE_KEYS = ("time_unit", "parts", "glue", "allow_missing", "components")
COMPONENT_KEYS = ("inputs", "outputs")
PART_KEYS = ("name", "file", "time_unit", "rename")


@dataclass(frozen=True)
class _Part:
    """One part of a recipe, as validated"""

    name: str
    file: str  # As written in the recipe
    path: Path
    time_unit: str
    rename: dict[str, str]


class _Errors:
    """The messages collected so far, and whether one of them stops the composition: a part
    that cannot be loaded or used, a time unit that is not one, or a glue file that cannot
    be read. The others are reported with everything found after them."""

    def __init__(self) -> None:
        self.messages: list[str] = []
        self.fatal = False

    def add(self, message: str, *, fatal: bool = False) -> None:
        self.messages.append(message)
        self.fatal = self.fatal or fatal


def _is_time(unit: str) -> bool:
    try:
        return bool(ureg.Quantity(1, unit).check("[time]"))
    except Exception:  # pint raises many kinds of errors on text that is not a unit
        return False


def _one_line(error: Exception) -> str:
    return " ".join(str(error).splitlines())


def _load_errors() -> tuple[type[Exception], ...]:
    """The errors that loading a part's file raises for a bad file"""
    errors: tuple[type[Exception], ...] = (exceptions.GotranxError, lark.exceptions.LarkError)
    try:
        import myokit
    except ImportError:  # pragma: no cover
        return errors
    return (*errors, myokit.MyokitError)


def _is_names(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(n, str) for n in value)


def _validate_components(tables: Any, errors: _Errors) -> None:
    if tables is None:
        return
    if not isinstance(tables, Mapping):
        errors.add("components must be a table of tables, one per component")
        return
    for name, table in tables.items():
        label = f"components.{name}"
        if not isinstance(table, Mapping):
            errors.add(f"{label} must be a table with inputs and outputs")
            continue
        for key in table:
            if key not in COMPONENT_KEYS:
                errors.add(
                    f"{label}: unknown key {key}: a component's keys are {_listing(COMPONENT_KEYS)}"
                )
        for key in COMPONENT_KEYS:
            if key not in table:
                errors.add(f"{label} has no {key}")
            elif not _is_names(table[key]):
                errors.add(f"{label}: {key} must be a list of names")


def _split_names(
    tables: Mapping[str, Mapping[str, Any]], applied: _Glue, errors: _Errors
) -> dict[str, tuple[list[str], list[str]]]:
    """Check the qualifiers of the components tables against the parts, and unqualify"""
    result: dict[str, tuple[list[str], list[str]]] = {}
    for name, table in tables.items():
        if not isinstance(table, Mapping) or not all(
            _is_names(table.get(k)) for k in COMPONENT_KEYS
        ):
            continue  # Reported by _validate_components
        lists: dict[str, list[str]] = {}
        for key in COMPONENT_KEYS:
            lists[key] = []
            for written in table[key]:
                part, dot, bare = written.partition(".")
                if not dot:
                    lists[key].append(written)
                elif part not in applied.own:
                    errors.add(f"components.{name}: {key}: {written}: {part} is not a part")
                elif bare not in applied.names(part):
                    msg = applied.not_a_name(bare, part)
                    errors.add(f"components.{name}: {key}: {msg}")
                else:
                    lists[key].append(bare)
        result[name] = (lists["inputs"], lists["outputs"])
    return result


def _validate(recipe: Mapping[str, Any], base: Path, errors: _Errors) -> list[_Part]:
    """Check the recipe's keys and values, and return its valid parts"""
    for key in recipe:
        if key not in RECIPE_KEYS:
            errors.add(f"unknown recipe key {key}: the keys are {_listing(RECIPE_KEYS)}")
    if "time_unit" not in recipe:
        errors.add("the recipe has no time_unit", fatal=True)
    elif not isinstance(recipe["time_unit"], str):
        errors.add("time_unit must be a string", fatal=True)
    elif not _is_time(recipe["time_unit"]):
        errors.add(f"time_unit {recipe['time_unit']} is not a unit of time", fatal=True)
    if not isinstance(recipe.get("glue", ""), str):
        errors.add("glue must be a string")
    if not isinstance(recipe.get("allow_missing", False), bool):
        errors.add("allow_missing must be true or false")

    _validate_components(recipe.get("components"), errors)

    raw_parts = recipe.get("parts")
    if not raw_parts:
        errors.add("the recipe has no parts", fatal=True)
        return []
    if not isinstance(raw_parts, list):
        errors.add("parts must be a list of tables, one per part", fatal=True)
        return []

    parts = []
    for i, part in enumerate(raw_parts, start=1):
        if not isinstance(part, Mapping):
            errors.add(f"part {i} must be a table", fatal=True)
            continue
        name = part.get("name")
        # Glue writes part.name, so a part's name must be a name
        valid_name = isinstance(name, str) and VALID_NAME.fullmatch(name) is not None
        label = f"part {name}" if valid_name else f"part {i}"
        for key in part:
            if key not in PART_KEYS:
                errors.add(f"{label}: unknown key {key}: a part's keys are {_listing(PART_KEYS)}")
        if "name" not in part:
            errors.add(f"part {i} has no name", fatal=True)
        elif not isinstance(name, str):
            errors.add(f"part {i}: name must be a string", fatal=True)
        elif not valid_name:
            errors.add(
                f"part {i}: name {name!r} is not a valid name: use letters, digits and "
                "underscores, not starting with a digit",
                fatal=True,
            )
        for key in ("file", "time_unit"):
            if key not in part:
                errors.add(f"{label} has no {key}", fatal=True)
            elif not isinstance(part[key], str):
                errors.add(f"{label}: {key} must be a string", fatal=True)
        unit = part.get("time_unit")
        if isinstance(unit, str) and not _is_time(unit):
            errors.add(f"{label}: time_unit {unit} is not a unit of time", fatal=True)
        renames = part.get("rename", {})
        if not (
            isinstance(renames, Mapping)
            and all(isinstance(k, str) and isinstance(v, str) for k, v in renames.items())
        ):
            errors.add(f"{label}: rename must be a table of names")
            renames = {}
        file = part.get("file")
        if valid_name and isinstance(file, str) and isinstance(unit, str):
            assert isinstance(name, str)
            parts.append(_Part(name, file, base / file, unit, dict(renames)))

    counts: dict[str, int] = defaultdict(int)
    for p in parts:
        counts[p.name] += 1
    for name, count in counts.items():
        if count > 1:
            errors.add(f"part names must be unique: {count} parts are named {name}", fatal=True)
    return parts


def _load_part(part: _Part) -> ODE:
    """The part's model, named after the part, so that messages about it name the part"""
    if part.path.suffix == ".cellml":
        from .myokit import cellml_to_gotran

        ode = cellml_to_gotran(part.path)
        # Re-parse, so that the atoms are those of an .ode file
        return rebuild(all_atoms(ode), name=part.name, comments=ode.comments)
    ode = load_ode(part.path, remove_singularities=False)
    return ODE(components=ode.components, t=ode.t, name=part.name, comments=ode.comments)


def _load_glue(path: Path, glue: str, errors: _Errors) -> tuple[GlueStatement, ...]:
    """The statements of the glue file, or none, with its error added to ``errors``.

    The parser stops at the first error, so a glue file reports one parse error at a time.
    """
    try:
        return load_glue(path)
    except (
        exceptions.UnknownStatement,
        exceptions.InvalidGlueBlock,
        exceptions.GlueOnlyConstruct,
    ) as e:
        errors.add(f"{glue}:{e.line}: {e}", fatal=True)
    except exceptions.MissingSymbolError as e:  # A name in a parameter value
        errors.add(f"{glue}:{e.line_no}: parameter values must be numbers", fatal=True)
    except exceptions.InvalidTreeError as e:
        if e.tree.data != "qualified_variable":  # pragma: no cover
            raise
        # `part.name` in a parameter value
        line = getattr(e.tree.meta, "line", "?")
        errors.add(f"{glue}:{line}: parameter values must be numbers", fatal=True)
    except lark.exceptions.UnexpectedInput as e:
        errors.add(f"{glue}:{e.line}: syntax error at column {e.column}", fatal=True)
    except UnicodeDecodeError as e:
        errors.add(_not_utf8(glue, e), fatal=True)
    return ()


def _kind(atom: atoms.Atom) -> str:
    if isinstance(atom, atoms.State):
        return "a state"
    if isinstance(atom, atoms.Parameter):
        return "a parameter"
    if isinstance(atom, atoms.StateDerivative):
        return "a state derivative"
    return "an intermediate"


def _already(name: str, atom: atoms.Atom, part: str) -> str:
    """The message for a new name that is already a name of the part"""
    kind = _kind(atom)
    if isinstance(atom, atoms.State):
        return f"{name} is already a state of {part}: use it"
    if isinstance(atom, atoms.StateDerivative):
        return f"{name} is already {kind} of {part}: use it, or write `{name} += ...`"
    return f"{name} is already {kind} of {part}: use it, or write `replace {name} = ...`"


class _Glue:
    """Applies the statements of a glue file to the parts, by the spec's rules table"""

    def __init__(
        self,
        statements: tuple[GlueStatement, ...],
        own: dict[str, dict[str, atoms.Atom]],
    ) -> None:
        self.own = own  # Each part's atoms by name, before the glue
        self.new: dict[str, dict[str, GlueStatement]] = {p: {} for p in own}
        self.defined_at: dict[tuple[str, str], int] = {}  # Glue definitions: their lines
        self.missing: dict[str, int] = {}  # Names defined nowhere: the first line using each
        self.errors: list[tuple[int, str]] = []
        self.accepted: list[GlueStatement] = []

        unknown: set[str] = set()
        statements_ = []
        for s in statements:
            if s.component in own:
                statements_.append(s)
            elif s.component not in unknown:
                unknown.add(s.component)
                self.errors.append((s.line, f"there is no part {s.component}"))
        # New names first, so that a name can be used before the line that defines it
        new_ok = [s.kind == "new" and self._check_new(s) for s in statements_]
        for s, ok in zip(statements_, new_ok):
            if s.kind == "replace":
                ok = self._check_replace(s)
            elif s.kind == "add":
                ok = self._check_add(s)
            ok = self._check_references(s) and ok
            if ok:
                self.accepted.append(s)

    def names(self, part: str) -> set[str]:
        return self.own[part].keys() | self.new[part].keys()

    def owner(self, name: str, exclude: str | None = None) -> str | None:
        """The first part, other than ``exclude``, that has ``name``"""
        return next((p for p in self.own if p != exclude and name in self.names(p)), None)

    def not_a_name(self, name: str, part: str) -> str:
        owner = self.owner(name, exclude=part)
        hint = f" (it is {owner}'s)" if owner is not None else ""
        return f"{name} is not a name of {part}{hint}"

    def error(self, s: GlueStatement, message: str) -> bool:
        self.errors.append((s.line, message))
        return False

    def _check_new(self, s: GlueStatement) -> bool:
        part, name = s.component, s.name
        if name in TIME_NAMES:
            return self.error(s, f"{name} is the name of time: choose another name")
        if (part, name) in self.defined_at:
            return self.error(
                s, f"{name} is already defined in the glue, at line {self.defined_at[part, name]}"
            )
        if name in self.own[part]:
            return self.error(s, _already(name, self.own[part][name], part))
        owner = self.owner(name, exclude=part)
        if owner is not None:
            return self.error(
                s,
                f"names must be unique: {name} is defined in {owner}; "
                "rename it in the recipe or choose another",
            )
        derivative = STATE_DERIV_EXPR.match(name)
        if s.block == "expressions" and derivative:  # The parser would look for the state
            state = derivative["state"]
            return self.error(
                s,
                f"{name} names the derivative of {state}, which is not a state of {part}: "
                "choose another name",
            )
        self.new[part][name] = s
        self.defined_at[part, name] = s.line
        return True

    def _check_replace(self, s: GlueStatement) -> bool:
        part, name = s.component, s.name
        if (part, name) in self.defined_at:
            return self.error(
                s, f"{name} is already defined in the glue, at line {self.defined_at[part, name]}"
            )
        target = self.own[part].get(name)
        if target is None:
            return self.error(s, self.not_a_name(name, part))
        if isinstance(target, atoms.State):
            return self.error(s, f"{name} is a state of {part}: a state cannot be replaced")
        if s.block == "parameters" and not isinstance(target, atoms.Parameter):
            return self.error(
                s, f"{name} is {_kind(target)} of {part}: replace it in an expressions block"
            )
        self.defined_at[part, name] = s.line
        return True

    def _check_add(self, s: GlueStatement) -> bool:
        if isinstance(self.own[s.component].get(s.name), atoms.StateDerivative):
            return True
        owner = self.owner(s.name, exclude=s.component)
        hint = f" (it is {owner}'s)" if owner is not None else ""
        return self.error(s, f"{s.name} is not the derivative of a state of {s.component}{hint}")

    def _check_references(self, s: GlueStatement) -> bool:
        ok = True
        for part, name in sorted(s.references, key=lambda r: (r[0] or "", r[1])):
            if part is None:
                if name in self.names(s.component) or name in TIME_NAMES:
                    continue
                if self.owner(name) is not None:
                    ok = self.error(s, self.not_a_name(name, s.component))
                else:
                    self.missing.setdefault(name, s.line)
            elif part not in self.own:
                ok = self.error(s, f"there is no part {part} (in {part}.{name})")
            elif name not in self.names(part):
                ok = self.error(s, self.not_a_name(name, part))
        return ok

    def apply(self) -> dict[str, dict[str, atoms.Atom]]:
        """Each part's atoms by name, after the accepted statements"""
        result = {p: dict(a) for p, a in self.own.items()}
        names = set(self.missing).union(*(self.names(p) for p in self.own))
        symbols: dict[str, Any] = {n: make_symbol(n) for n in names}
        t = sp.Symbol("t")  # As the parser has it
        symbols.update({n: t for n in TIME_NAMES})

        def expr(s: GlueStatement) -> tuple[atoms.Expression, sp.Expr]:
            assert s.atom.value is not None and not isinstance(s.atom, atoms.Parameter)
            tree = unqualify(s.atom.value.tree)
            return atoms.Expression(tree=tree), build_expression(tree, symbols)

        # Definitions first, then the terms added to derivatives
        for s in sorted(self.accepted, key=lambda s: s.kind == "add"):
            part, name = s.component, s.name
            if s.kind == "add":
                d = result[part][name]
                assert isinstance(d, atoms.StateDerivative)
                result[part][name] = attr.evolve(d, expr=d.expr + expr(s)[1])
            elif s.block == "parameters":
                old = self.own[part].get(name)
                if old is None:
                    result[part][name] = s.atom
                else:
                    # A new declaration keeps the unit and description it does not give,
                    # as a replace in an expressions block keeps the unit
                    given_unit = s.atom.unit_str is not None
                    result[part][name] = attr.evolve(
                        s.atom,
                        unit_str=s.atom.unit_str if given_unit else old.unit_str,
                        unit=s.atom.unit if given_unit else old.unit,
                        description=s.atom.description or old.description,
                    )
            else:
                assert isinstance(s.atom, atoms.Assignment)
                value, e = expr(s)
                old = self.own[part].get(name)
                unit_str = s.atom.unit_str
                if unit_str is None and old is not None:
                    # An input keeps the parameter's unit; a trailing comment is not a unit
                    unit_str = old.unit_str
                new: atoms.Assignment = atoms.Intermediate(
                    name=name,
                    value=value,
                    expr=e,
                    components=(part,),
                    unit_str=unit_str,
                    comment=s.atom.comment,
                )
                if isinstance(old, atoms.StateDerivative):
                    new = new.to_state_derivative(old.state)
                result[part][name] = new
        return result


def _in_component(atom: atoms.Atom, part: str) -> atoms.Atom:
    if isinstance(atom, atoms.StateDerivative):
        return attr.evolve(
            atom, components=(part,), state=attr.evolve(atom.state, components=(part,))
        )
    return attr.evolve(atom, components=(part,))


def _rename(part: _Part, ode: ODE, errors: _Errors) -> ODE:
    """The part's model with its renames, and a message for each rename refused. The
    others are still made, so that the checks that follow see the names the recipe
    means."""
    if not part.rename:
        return ode
    try:
        return rename(ode, part.rename)
    except exceptions.RenameError as e:
        for message in e.messages:
            errors.add(f"part {part.name}: {message}")
        rest = {old: new for old, new in part.rename.items() if old not in e.problems}
        return rename(ode, rest) if rest else ode


def _circular(
    result: dict[str, dict[str, atoms.Atom]], glue: _Glue
) -> tuple[int | None, str] | None:
    """The line of the first glue statement on a cycle among the equations, if any, and
    a message naming the cycle"""
    sorter: graphlib.TopologicalSorter = graphlib.TopologicalSorter()
    for by_name in result.values():
        for name in sorted(by_name):
            atom = by_name[name]
            if isinstance(atom, atoms.Assignment):
                sorter.add(name, *sorted(str(s) for s in atom.expr.free_symbols))
    try:
        sorter.prepare()
    except graphlib.CycleError as e:
        # Each node of the cycle is used by the next, so reversed, each uses the next
        cycle = list(reversed(e.args[1][1:]))
        lines: dict[str, int] = {}
        for s in glue.accepted:  # In the file's order, so each name gets its first line
            lines.setdefault(s.name, s.line)
        first = min(range(len(cycle)), key=lambda i: (lines.get(cycle[i], math.inf), cycle[i]))
        cycle = cycle[first:] + cycle[:first]
        uses = " uses ".join([*cycle, cycle[0]])
        return lines.get(cycle[0]), f"{uses}: the equations are circular"
    return None


def _share_parameters(atoms_: dict[str, dict[str, atoms.Atom]]) -> None:
    """Give each parameter the components ``(home, *every other part that uses it)``"""
    users: dict[str, set[str]] = defaultdict(set)
    for part, by_name in atoms_.items():
        for atom in by_name.values():
            if isinstance(atom, atoms.Assignment):
                for symbol in atom.expr.free_symbols:
                    users[str(symbol)].add(part)
    for part, by_name in atoms_.items():
        for name, atom in by_name.items():
            if isinstance(atom, atoms.Parameter):
                others = tuple(p for p in atoms_ if p != part and p in users[name])
                if others:
                    by_name[name] = attr.evolve(atom, components=(part, *others))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _Recipe:
    """A recipe, as read"""

    data: Mapping[str, Any]
    base: Path  # What relative paths resolve against
    file: Path | None = None  # The recipe file, if it is one
    text: str | None = None  # Its text


def _not_utf8(name: str, error: UnicodeDecodeError) -> str:
    return f"{name}: not UTF-8 ({error.reason} at position {error.start})"


def _read_recipe(recipe: str | Path | Mapping[str, Any], base_dir: str | Path | None) -> _Recipe:
    if isinstance(recipe, Mapping):
        return _Recipe(recipe, Path.cwd() if base_dir is None else Path(base_dir))
    recipe_file = Path(recipe)
    if not recipe_file.is_file():
        raise exceptions.CompositionError([f"recipe file {recipe_file} not found"])
    try:
        text = recipe_file.read_text(encoding="utf-8")  # TOML is UTF-8, whatever the locale
        data = tomllib.loads(text)
    except UnicodeDecodeError as e:
        raise exceptions.CompositionError([_not_utf8(recipe_file.name, e)]) from e
    except tomllib.TOMLDecodeError as e:
        raise exceptions.CompositionError([f"{recipe_file.name}: {e}"]) from e
    return _Recipe(data, recipe_file.parent, recipe_file, text)


def input_files(
    recipe: str | Path | Mapping[str, Any], *, base_dir: str | Path | None = None
) -> list[tuple[str, Path]]:
    """The files composing ``recipe`` reads, each with what it is
    (``"the recipe"``, ``"the glue file"`` or ``"the file of part <name>"``), as far as
    the recipe names them. Nothing is checked: a recipe that cannot be read gives the
    recipe file alone, and :func:`compose` says what is wrong with it.

    Parameters are as for :func:`compose`.
    """
    try:
        read = _read_recipe(recipe, base_dir)
    except exceptions.CompositionError:
        return [] if isinstance(recipe, Mapping) else [("the recipe", Path(recipe))]
    files = [] if read.file is None else [("the recipe", read.file)]
    glue = read.data.get("glue")
    if isinstance(glue, str):
        files.append(("the glue file", read.base / glue))
    parts = read.data.get("parts")
    for i, part in enumerate(parts if isinstance(parts, list) else [], start=1):
        if isinstance(part, Mapping) and isinstance(part.get("file"), str):
            name = part.get("name")
            label = name if isinstance(name, str) else str(i)
            files.append((f"the file of part {label}", read.base / part["file"]))
    return files


def compose(
    recipe: str | Path | Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
    remove_singularities: bool = True,
) -> ODE:
    """Compose an ODE from the parts and the glue file a recipe names

    Parameters
    ----------
    recipe : str | Path | Mapping[str, Any]
        A recipe (TOML) file, or a dict with the same keys. ``time_unit`` and ``parts``
        are required, ``glue`` and ``allow_missing`` (default false) optional. Each part
        has a ``name``, a ``file`` (``.ode`` or ``.cellml``) and a ``time_unit``, and may
        ``rename`` its names (``{old = "new"}``)
    base_dir : str | Path | None, optional
        For a dict, the directory relative paths resolve against, by default the current
        directory. A recipe file's relative paths resolve against its own directory.
    remove_singularities : bool, optional
        Guard the composed model's removable singularities, by default True, as
        :func:`gotranx.load.load_ode` does. The parts are loaded without, so this is
        done once, on the composed model.

    Returns
    -------
    gotranx.ode.ODE
        The composed model, in which each part is one component, named after the part.
        Its comments hold the recipe and the sha256 of every file read, and no timestamp,
        so that composing the same files again saves the same text.

    Raises
    ------
    exceptions.CompositionError
        With every message, if anything is wrong. A glue message starts with
        ``<glue file>:<line>:``.

    Notes
    -----
    A part whose ``time_unit`` differs from the model's has every state derivative
    multiplied by ``k``, one model time unit in the part's unit, and ``t`` replaced by
    ``k*t``. The glue is in the model's time unit.

    With ``allow_missing = true``, glue may use names that no part defines; they are the
    model's missing variables. Such a model can be saved, but gotranx cannot load an
    ``.ode`` file that uses names it does not define.

    ``components = {<name> = {inputs = [...], outputs = [...]}}`` calls
    :meth:`gotranx.ode.ODE.split` on the composed model, one table after another in recipe
    order. Names may be written ``part.name``, which is checked against the part and then
    unqualified. A split that fails is raised as a ``CompositionError`` with its messages,
    each starting ``components.<name>:``.
    """
    read = _read_recipe(recipe, base_dir)
    data, base, recipe_file = read.data, read.base, read.file
    source = "a dict" if recipe_file is None else recipe_file.name
    name = "composed" if recipe_file is None else recipe_file.stem

    # 1. The recipe
    errors = _Errors()
    parts = _validate(data, base, errors)
    glue: str | None = data.get("glue") if isinstance(data.get("glue"), str) else None
    allow_missing = data.get("allow_missing", False) is True

    # 2. The parts and the glue
    odes: dict[str, ODE] = {}
    for part in parts:
        if not part.path.is_file():
            errors.add(f"part {part.name}: file {part.file} not found", fatal=True)
            continue
        try:
            odes[part.name] = _load_part(part)
        except lark.exceptions.UnexpectedInput as e:
            where = f"{part.file}:{e.line}"
            errors.add(f"part {part.name}: {where}: syntax error at column {e.column}", fatal=True)
        except _load_errors() as e:
            errors.add(f"part {part.name}: {part.file}: {_one_line(e)}", fatal=True)
    statements: tuple[GlueStatement, ...] = ()
    if glue is not None:
        if (base / glue).is_file():
            statements = _load_glue(base / glue, glue, errors)
        else:
            errors.add(f"glue file {glue} not found", fatal=True)
    # Only what stops the later steps stops here; the rest is reported with what they find
    if errors.fatal:
        raise exceptions.CompositionError(errors.messages)

    # 3. Rename, then rescale time. 4. One component per part.
    own: dict[str, dict[str, atoms.Atom]] = {}
    for part in parts:
        ode = _rename(part, odes[part.name], errors)
        k = ureg.Quantity(1, data["time_unit"]).to(part.time_unit).magnitude
        if k != 1:
            ode = rescale_time(ode, k)
        own[part.name] = {a.name: _in_component(a, part.name) for a in all_atoms(ode)}

    # 5. Names are unique across parts
    owners: dict[str, list[str]] = defaultdict(list)
    for part_name, by_name in own.items():
        for n in by_name:
            owners[n].append(part_name)
    duplicates = {n: o for n, o in owners.items() if len(o) > 1}
    for n, o in sorted(duplicates.items()):
        atom = own[o[0]][n]
        if isinstance(atom, atoms.StateDerivative) and atom.state.name in duplicates:
            continue  # Reported for the state
        errors.add(
            f"names must be unique: {n} is defined in {' and in '.join(o)}; rename it in the recipe"
        )

    # 6. The glue. 7. Parameters used by several parts are under each.
    applied = _Glue(statements, own)
    for line, message in sorted(applied.errors):
        errors.add(f"{glue}:{line}: {message}")
    # 8. Names used but defined nowhere
    if applied.missing and not allow_missing:
        where = ", ".join(f"{n} ({glue}:{line})" for n, line in sorted(applied.missing.items()))
        errors.add(f"used but defined nowhere: {where}; define them, or set allow_missing = true")
    tables = data.get("components") or {}
    splits = _split_names(tables, applied, errors) if isinstance(tables, Mapping) else {}
    if errors.messages:
        raise exceptions.CompositionError(errors.messages)
    result = applied.apply()
    circular = _circular(result, applied)
    if circular is not None:
        at, message = circular
        where = "" if at is None else f"{glue}:{at}: "
        raise exceptions.CompositionError([f"{where}{message}"])
    _share_parameters(result)

    # 9. The header: the recipe and the sha256 of every file read, and no timestamp. Each
    # line is saved as it is, so that the recipe can be read back from it.
    recipe_text = read.text
    if recipe_text is None:
        recipe_text = json.dumps(dict(data), sort_keys=True, indent=1, default=str)
    header = [f"Composed by gotranx {version('gotranx')} from {source}"]
    header += recipe_text.splitlines()
    files = [] if recipe_file is None else [(recipe_file.name, recipe_file)]
    if glue is not None:
        files.append((glue, base / glue))
    files += [(part.file, part.path) for part in parts]
    header += [f"sha256 {file}: {_sha256(path)}" for file, path in files]

    # 10. Build
    all_ = [atom for by_name in result.values() for atom in by_name.values()]
    # gotranx cannot parse a name it has no definition of, so each missing one is a
    # parameter of a component of its own while the model is built, and then dropped
    placeholder = "missing"
    while placeholder in result:
        placeholder += "_"
    missing = [
        atoms.Parameter(name=n, value=0.0, components=(placeholder,)) for n in applied.missing
    ]
    composed = rebuild(
        [*all_, *missing],
        name=name,
        comments=[atoms.Comment(line, verbatim=True) for line in header],
    )
    if missing:
        composed = ODE(
            components=tuple(c for c in composed.components if c.name != placeholder),
            t=composed.t,
            name=composed.name,
            comments=composed.comments,
        )
    # 11. The components tables, in recipe order, before the guards: split follows each
    # assignment's parse tree, which is that of the unguarded expression. A SplitError is
    # re-raised as a CompositionError with the same messages, each saying which table it
    # came from
    for component, (inputs, outputs) in splits.items():
        try:
            composed = composed.split(component, inputs=inputs, outputs=outputs)
        except exceptions.SplitError as e:
            raise exceptions.CompositionError(
                [f"components.{component}: {m}" for m in e.messages]
            ) from e
    # 12. Guard the singularities
    if remove_singularities:
        composed = composed.remove_singularities()
    return composed
