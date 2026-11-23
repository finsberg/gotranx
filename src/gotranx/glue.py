"""Glue files: how the parts of a composition are coupled.

A glue file is written in the ``.ode`` grammar, in blocks that each name exactly
one component (the part the block belongs to), plus three constructs that only
glue files accept:

- ``part.name``, which references a name of another part;
- ``replace name = ...``, which changes an existing name of the part, in a
  ``parameters`` or an ``expressions`` block;
- ``dX_dt += ...``, which adds a term to a state derivative.

A glue file cannot declare states. This module only parses a glue file into
:class:`GlueStatement` objects; what each statement does to the parts is up to
the composition that applies it.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lark

from . import atoms
from . import exceptions
from .parser import Parser
from .transformer import (
    TreeToODE,
    TWO_WORD_STATEMENTS,
    check_statement_word,
    find_assignments,
    tree2parameter,
)

Kind = Literal["new", "replace", "add"]
Block = Literal["parameters", "expressions"]


@dataclass(frozen=True)
class GlueStatement:
    """One statement of a glue file

    Attributes
    ----------
    kind : "new", "replace" or "add"
        ``name = ...``, ``replace name = ...`` or ``dX_dt += ...``
    block : "parameters" or "expressions"
        The kind of block the statement is in
    component : str
        The block's component, i.e. the part the statement belongs to
    name : str
        The name the statement defines, replaces or (for "add") adds to,
        e.g. ``"dcai_dt"``
    atom : atoms.Parameter | atoms.Assignment
        The statement as an atom in ``component``. For "add", an assignment of
        the added term to ``name``. Its expression tree may hold ``part.name``
        (``qualified_variable``) nodes, which :func:`unqualify` replaces
    references : frozenset[tuple[str | None, str]]
        ``(part, name)`` for every name the right-hand side uses, with part
        ``None`` for a bare name
    line : int
        The line of the glue file the statement starts on
    """

    kind: Kind
    block: Block
    component: str
    name: str
    atom: atoms.Parameter | atoms.Assignment
    references: frozenset[tuple[str | None, str]]
    line: int


def references(tree: lark.Tree) -> frozenset[tuple[str | None, str]]:
    """``(part, name)`` for every name used in an expression tree, with part
    ``None`` for a bare name"""
    refs: set[tuple[str | None, str]] = set()
    for subtree in tree.iter_subtrees():
        if subtree.data == "variable":
            refs.add((None, str(subtree.children[0])))
        elif subtree.data == "qualified_variable":
            part, name = subtree.children
            refs.add((str(part), str(name)))
    return frozenset(refs)


def glue_statement(tree: lark.Tree, block: Block, component: str) -> GlueStatement:
    """Convert the tree of one statement of a glue file to a :class:`GlueStatement`

    Parameters
    ----------
    tree : lark.Tree
        The statement, as parsed
    block : "parameters" or "expressions"
        The kind of block the statement is in
    component : str
        The component the block names

    Returns
    -------
    GlueStatement
        The statement
    """
    check_statement_word(tree)
    kind: Kind = "new"
    children = tree.children
    if tree.data in TWO_WORD_STATEMENTS:
        kind = "replace"
        children = children[1:]  # Drop the word `replace`
    elif tree.data == "add_assignment":
        kind = "add"

    # The tree without `replace`, as the plain statement would have parsed
    atom: atoms.Parameter | atoms.Assignment
    if block == "parameters":
        data = "scalarparam" if tree.data.endswith("scalarparam") else "param"
        atom = tree2parameter(
            lark.Tree(data, children, meta=tree.meta),
            components=(component,),
            cls=atoms.Parameter,
        )
    else:
        (atom,) = find_assignments(
            lark.Tree("assignment", children, meta=tree.meta), components=(component,)
        )

    value = children[1]
    assert isinstance(value, lark.Tree)  # The grammar's atoms are all rules
    return GlueStatement(
        kind=kind,
        block=block,
        component=component,
        name=str(children[0]),
        atom=atom,
        references=references(value),
        line=getattr(tree.children[0], "line", 0),
    )


def parse_glue(text: str) -> tuple[GlueStatement, ...]:
    """Parse the text of a glue file

    Parameters
    ----------
    text : str
        The glue file's text

    Returns
    -------
    tuple[GlueStatement, ...]
        Its statements, in the order of the file

    Raises
    ------
    exceptions.InvalidGlueBlock
        For a ``states`` block, or a block that does not name exactly one component
    exceptions.UnknownStatement
        For ``<word> name = ...`` with a word other than ``replace``
    """
    parser = Parser(parser="lalr", transformer=TreeToODE(glue=True), propagate_positions=True)
    return typing.cast(tuple[GlueStatement, ...], parser.parse(text))


def load_glue(path: str | Path) -> tuple[GlueStatement, ...]:
    """Load a glue file

    Parameters
    ----------
    path : str | Path
        Path to the file

    Returns
    -------
    tuple[GlueStatement, ...]
        Its statements, in the order of the file

    Raises
    ------
    exceptions.ODEFileNotFound
        If the file is not found
    """
    fname = Path(path)
    if not fname.is_file():
        raise exceptions.ODEFileNotFound(fname)
    return parse_glue(fname.read_text())


def unqualify(tree: lark.Tree) -> lark.Tree:
    """Replace each ``part.name`` node of an expression tree with a plain
    variable node ``name``, which keeps the line number

    Parameters
    ----------
    tree : lark.Tree
        An expression tree, e.g. ``GlueStatement.atom.value.tree``

    Returns
    -------
    lark.Tree
        A new tree; ``tree`` itself is left unchanged
    """
    if tree.data == "qualified_variable":
        _part, name = tree.children
        return lark.Tree("variable", [name], meta=tree.meta)
    return lark.Tree(
        tree.data,
        [unqualify(c) if isinstance(c, lark.Tree) else c for c in tree.children],
        meta=tree.meta,
    )
