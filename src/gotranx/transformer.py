from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple
from typing import TypeVar, Any, TYPE_CHECKING

import lark

from . import atoms
from . import exceptions
from . import ode_component
from . import units

if TYPE_CHECKING:
    from .glue import Block, GlueStatement

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
            except Exception:
                # Not a proper unit so it's a comment. On free text pint raises
                # UndefinedUnitError, but also tokenize.TokenError (an apostrophe,
                # as in "Land's") or AssertionError ("?").
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
    s: list[Any],
) -> tuple[int, tuple[str, ...]]:
    """Find components in a list

    Parameters
    ----------
    s : list[Any]
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
    s: list[Any],
    cls: type[T],
) -> tuple[T | atoms.Comment, ...]:
    """Convert a list of lark trees to parameters

    Parameters
    ----------
    s : list[Any]
        The list
    cls : type[T]
        The class of the parameters

    Returns
    -------
    tuple[T | atoms.Comment, ...]
        The parameters and any standalone comments
    """
    i, components = find_components(s)
    res: list[T | atoms.Comment] = []
    for p in s[i:]:
        if isinstance(p, lark.Tree):
            res.append(tree2parameter(p, components=components, cls=cls))
        elif isinstance(p, atoms.Comment):
            res.append(p)
    return tuple(res)


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


# Statements of the form `<word> name = ...`, valid only with <word> == "replace"
TWO_WORD_STATEMENTS = ("replace_param", "replace_scalarparam", "replace_assignment")
# Statements only glue files accept
GLUE_STATEMENTS = (*TWO_WORD_STATEMENTS, "add_assignment")


def check_statement_word(tree: lark.Tree) -> None:
    """Raise :class:`exceptions.UnknownStatement` if ``tree`` is a two-word
    statement whose first word is not ``replace``.

    The grammar reads any two names before ``=``, so that ``replace`` stays
    an ordinary name everywhere else.
    """
    if tree.data in TWO_WORD_STATEMENTS:
        word = tree.children[0]
        if word != "replace":
            raise exceptions.UnknownStatement(word=str(word), line=getattr(word, "line", 0))


def refuse_glue_constructs(items: list[Any]) -> None:
    """Raise if a block of a plain ``.ode`` file holds a glue-only construct

    Parameters
    ----------
    items : list[Any]
        The children of a ``states``, ``parameters`` or ``expressions`` block

    Raises
    ------
    exceptions.UnknownStatement
        For a two-word statement that does not start with ``replace``
    exceptions.GlueOnlyConstruct
        For ``replace``, ``+=`` or ``part.name``
    """
    for item in items:
        if not isinstance(item, lark.Tree):
            continue
        check_statement_word(item)
        if item.data in GLUE_STATEMENTS or any(
            t.data == "qualified_variable" for t in item.iter_subtrees()
        ):
            raise exceptions.GlueOnlyConstruct()


def first_line(items: list[Any]) -> int | None:
    """The line of the first token in ``items`` (searching trees depth first), if any"""
    for item in items:
        if isinstance(item, lark.Token):
            return item.line
        if isinstance(item, lark.Tree):
            line = first_line(item.children)
            if line is not None:
                return line
    return None


class TreeToODE(lark.Transformer):
    """Transform a lark tree to an ODE

    See https://lark-parser.readthedocs.io/en/latest/recipes.html
    for more information on how to use lark transformers.

    Parameters
    ----------
    glue : bool, optional
        Transform a glue file instead, by default False. The result is then a
        tuple of :class:`gotranx.glue.GlueStatement`, in the order of the file
        (see :func:`gotranx.glue.parse_glue`). Without it, ``replace``, ``+=``
        and ``part.name`` raise :class:`exceptions.GlueOnlyConstruct`.
    """

    def __init__(self, glue: bool = False) -> None:
        super().__init__()
        self.glue = glue

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

    def states(self, s: list[Any]) -> tuple[atoms.State | atoms.Comment, ...]:
        """Convert a list of states to atoms.State"""
        if self.glue:
            raise exceptions.InvalidGlueBlock(
                "glue files cannot declare states", line=first_line(s)
            )
        refuse_glue_constructs(s)
        return lark_list_to_parameters(s, cls=atoms.State)

    def parameters(
        self, s: list[Any]
    ) -> tuple[atoms.Parameter | atoms.Comment, ...] | tuple[GlueStatement, ...]:
        """Convert a list of parameters to atoms.Parameter
        (or to glue statements, in glue mode)"""
        if self.glue:
            return self._glue_statements(s, block="parameters")
        refuse_glue_constructs(s)
        return lark_list_to_parameters(s, cls=atoms.Parameter)

    def expressions(
        self, s
    ) -> tuple[atoms.Assignment | atoms.Comment, ...] | tuple[GlueStatement, ...]:
        """Convert a list of expressions to atoms.Assignment and atoms.Comment
        (or to glue statements, in glue mode)"""
        if self.glue:
            return self._glue_statements(s, block="expressions")
        refuse_glue_constructs(s)
        i, components = find_components(s)

        assignments: list[atoms.Assignment | atoms.Comment] = []

        for si in s[i:]:
            if isinstance(si, atoms.Comment):
                assignments.append(si)
            else:
                assignments.extend(find_assignments(si, components=components))

        return tuple(assignments)

    def _glue_statements(self, s: list[Any], block: Block) -> tuple[GlueStatement, ...]:
        """The glue statements of one block, which must name exactly one component"""
        from .glue import glue_statement  # lazy: glue imports this module

        i, components = find_components(s)
        trees = [si for si in s[i:] if isinstance(si, lark.Tree)]
        if not trees:  # Only comments and blank lines
            return ()
        if len(components) != 1 or components[0] == "":
            names = ", ".join(f'"{c}"' for c in components if c) or "none"
            raise exceptions.InvalidGlueBlock(
                "a glue block must name exactly one component, the part it belongs to, "
                f"but this one names {names}",
                line=first_line(s),
            )
        return tuple(glue_statement(tree, block=block, component=components[0]) for tree in trees)

    def ode(self, s) -> LarkODE | tuple[GlueStatement, ...]:
        """Convert a lark tree to an ODE

        Parameters
        ----------
        s : list[Any]
            List of objects that could by states, parameters, assignments, or comments

        Returns
        -------
        LarkODE
            A tuple of components and comments (in glue mode, a tuple of
            the file's glue statements, in order)
        """
        if self.glue:
            return tuple(
                statement for block in s if isinstance(block, tuple) for statement in block
            )

        # FIXME: Could use Enum here
        mapping = {
            atoms.Parameter: "parameters",
            atoms.Assignment: "assignments",
            atoms.State: "states",
        }

        components: dict[str, dict[str, set[atoms.Atom]]] = defaultdict(
            lambda: {atom: set() for atom in mapping.values()},
        )

        comments = []
        for line in s:  # Each line in the block is now a tuple
            if isinstance(line, atoms.Comment):
                comments.append(line)
                continue
            if isinstance(line, str):
                assert line.strip() == "", f"Invalid line {line!r}"
                # Skip empty lines
                continue

            for atom in line:  # State, Parameters, Assignment, or standalone Comment
                if isinstance(atom, atoms.Comment):
                    comments.append(atom)
                    continue

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
