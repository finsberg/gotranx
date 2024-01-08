from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypeVar, Callable
from functools import reduce

from structlog import get_logger

from . import atoms
from .ode import ODE
from .codegen.python import PythonCodeGenerator

T = TypeVar("T")
logger = get_logger()


def break_comment_at_80(acc, x):
    if x == "#":
        return acc + "\n#"

    if len(acc.split("\n")[-1]) + len(x) > 80:
        return acc + "\n# " + x
    else:
        return acc + " " + x


class GotranODECodePrinter(PythonCodeGenerator):
    def print_comments(self) -> str:
        if len(self.ode.comments) == 0:
            return ""
        text = "# "

        for comment in self.ode.comments:
            text += reduce(break_comment_at_80, comment.text.split(" "))

        text += "\n\n"
        return text

    def print_states(self) -> str:
        d: dict[tuple[str, ...], list[atoms.State]] = defaultdict(list)
        for state in self.ode.states:
            d[state.components].append(state)

        text = ""
        for components, states in d.items():
            text += start_odeblock("states", names=components) + "\n"
            text += ", ".join([print_ScalarParam(s, doprint=self.printer.doprint) for s in states])
            text += "\n)\n\n"
        return text

    def print_parameters(self) -> str:
        d: dict[tuple[str, ...], list[atoms.Parameter]] = defaultdict(list)
        for parameter in self.ode.parameters:
            d[parameter.components].append(parameter)

        text = ""
        for components, parameters in d.items():
            text += start_odeblock("parameters", names=components) + "\n"
            text += ",\n".join(
                [print_ScalarParam(p, doprint=self.printer.doprint) for p in parameters]
            )
            text += "\n)\n\n"

        return text

    def print_assignments(self) -> str:
        d: dict[tuple[str, ...], list[atoms.Assignment]] = defaultdict(list)
        for i in self.ode.intermediates + self.ode.state_derivatives:
            d[i.components].append(i)

        text = ""
        for components, intermediates in d.items():
            text += start_odeblock("expressions", names=components, is_expression=True) + "\n"
            text += "\n".join([print_assignment(i) for i in intermediates])
            text += "\n\n"

        return text

    def format(self, code: str) -> str:
        return self._formatter(code)


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


def print_assignment(a: atoms.Assignment) -> str:
    s = f"{a.name} = {a.expr}"
    if a.unit_str is not None:
        s += f" # {a.unit_str}"
    return s


def write_ODE_to_ode_file(ode: ODE, path: Path) -> None:
    # Just make sure it is a Path object
    printer = GotranODECodePrinter(ode)
    path = Path(path)
    text = ""
    text += printer.print_comments()
    text += printer.print_states()
    text += printer.print_parameters()
    text += printer.print_assignments()
    path.write_text(text)
    logger.info(f"Wrote {path}")
