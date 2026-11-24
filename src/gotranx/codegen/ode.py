from __future__ import annotations

from collections import defaultdict
from typing import TypeVar, Callable
from functools import reduce
from itertools import groupby
from sympy.printing.str import StrPrinter
from structlog import get_logger

from .. import atoms
from ..ode import ODE
from .base import _print_Piecewise

T = TypeVar("T")
logger = get_logger()


def break_comment_at_80(acc, x):
    if x == "#":
        return acc + "\n#"

    if len(acc.split("\n")[-1]) + len(x) > 80:
        return acc + "\n# " + x
    else:
        return acc + " " + x


class BaseGotranODECodePrinter(StrPrinter):
    def _print_Mul(self, expr):
        # The parser reads `-2.5` as -1 times 2.5, unevaluated. sympy prints that as
        # `-1*2.5`, which reads back as (-1*1)*2.5, so each print and parse added a factor.
        # Printed as it was written, it reads back as itself.
        if len(expr.args) == 2 and expr.args[0] == -1:
            number = expr.args[1]
            if (number.is_Integer or number.is_Float) and number.is_nonnegative:
                return "-" + self._print(number)
        return super()._print_Mul(expr)

    def _print_Relational(self, expr):
        # v = super()._print_Relational(expr)
        lhs = self._print(expr.lhs)
        rhs = self._print(expr.rhs)

        relop2str = {
            "<": "Lt",
            "<=": "Le",
            ">": "Gt",
            ">=": "Ge",
            "==": "Eq",
            "!=": "Ne",
        }
        relop = relop2str[expr.rel_op]
        return f"{relop}({lhs}, {rhs})"

    def _print_Or(self, expr):
        return f"Or({', '.join(self._print(a) for a in expr.args)})"

    def _print_And(self, expr):
        return f"And({', '.join(self._print(a) for a in expr.args)})"

    def _print_BooleanFalse(self, expr):
        return "0"

    def _print_BooleanTrue(self, expr):
        return "1"

    def _print_Piecewise(self, expr):
        conds, exprs = _print_Piecewise(self, expr)

        result = []

        for c, e in zip(conds, exprs):
            result.append("Conditional(")
            result.append(f"{c}")
            result.append(", ")
            result.append(f"{e}")
            result.append(", ")

        if c == "1":
            result = result[:-6]
            result.append(f", {e}")
        else:
            raise ValueError("Last condition in Piecewise must be True")

        result.append(")" * (len(conds) - 1))
        return "".join(result)


class GotranODECodePrinter(BaseGotranODECodePrinter):
    def __init__(self, ode: ODE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ode = ode

    def print_comments(self) -> str:
        lines: list[str] = []
        for verbatim, group in groupby(self.ode.comments, key=lambda c: c.verbatim):
            if verbatim:  # One line each, as it is
                lines.extend(f"# {c.text}" if c.text else "#" for c in group)
                continue
            text = "\n".join(reduce(break_comment_at_80, c.text.split(" ")) for c in group)
            if len(text) > 0:
                lines.append("# " + "\n# ".join(text.strip().split("\n")))
        if not lines:
            return ""
        return "\n".join(lines) + "\n\n"

    def print_states(self) -> str:
        d: dict[tuple[str, ...], list[atoms.State]] = defaultdict(list)
        for state in self.ode.states:
            d[state.components].append(state)

        text = ""
        for components, states in d.items():
            text += start_odeblock("states", names=components) + "\n"
            text += ",\n".join([print_ScalarParam(s, doprint=self.doprint) for s in states])
            text += "\n)\n\n"
        return text

    def print_parameters(self) -> str:
        d: dict[tuple[str, ...], list[atoms.Parameter]] = defaultdict(list)
        for parameter in self.ode.parameters:
            d[parameter.components].append(parameter)

        text = ""
        for components, parameters in d.items():
            text += start_odeblock("parameters", names=components) + "\n"
            text += ",\n".join([print_ScalarParam(p, doprint=self.doprint) for p in parameters])
            text += "\n)\n\n"

        return text

    def print_assignments(self) -> str:
        d: dict[tuple[str, ...], list[atoms.Assignment]] = defaultdict(list)
        for i in self.ode.intermediates + self.ode.state_derivatives:
            d[i.components].append(i)

        # Unnamed expressions are printed without a header, so they go first: after a
        # named block they would be read back as part of it. (sorted is stable.)
        text = ""
        for components, intermediates in sorted(d.items(), key=lambda x: x[0] not in ((), ("",))):
            text += start_odeblock("expressions", names=components, is_expression=True) + "\n"
            text += "\n".join([print_assignment(i, doprint=self.doprint) for i in intermediates])
            text += "\n\n"

        return text


def start_odeblock(case: str, names: tuple[str, ...] = (), is_expression: bool = False):
    if len(names) == 0 or (len(names) == 1 and names[0] == ""):
        if is_expression:
            return ""
        return f"{case}("
    else:
        args = ", ".join(f'"{n}"' for n in names)

        if is_expression:
            return f"{case}({args})"
        else:
            return f"{case}({args},"


def join(*args: str) -> str:
    non_empty_args = [a for a in args if a != ""]
    if len(non_empty_args) == 0:
        return ""
    elif len(non_empty_args) == 1:
        return ", " + non_empty_args[0]
    else:
        return ", " + ", ".join(args)


def print_ScalarParam(p: atoms.Atom, doprint: Callable[[str], str]) -> str:
    unit_str = "" if p.unit_str is None else f'unit="{p.unit_str}"'
    description = "" if p.description is None else f'description="{p.description}"'
    if unit_str == "" and description == "":
        ret = f"{p.name}={doprint(p.value)}"  # type: ignore
    else:
        kwargs_str = join(unit_str, description)
        ret = f"{p.name}=ScalarParam({doprint(p.value)}{kwargs_str})"  # type: ignore

    logger.debug(f"Saving parameters {p} as {ret!r}")
    return ret


def print_assignment(a: atoms.Assignment, doprint: Callable[[str], str]) -> str:
    s = f"{a.name} = {doprint(a.expr)}"
    unit_or_comment = ""
    if a.comment is not None:
        unit_or_comment = a.comment.text
    if a.unit_str is not None and a.unit_str != "1":
        unit_or_comment = a.unit_str
    if unit_or_comment != "":
        s += f" # {unit_or_comment}"

    return s
