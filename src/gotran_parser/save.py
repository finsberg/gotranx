from collections import defaultdict
from pathlib import Path
from typing import Iterable
from typing import TypeVar

from structlog import get_logger

from . import atoms
from .ode import ODE

T = TypeVar("T")
logger = get_logger()


def start_odeblock(case, name=None, info=None):
    if name is None:
        return f"{case}("
    else:
        if info is None:
            return f'{case}("{name}",'
        else:
            return f'{case}("{name}", "{info}",'


def print_ScalarParam(p: atoms.Atom) -> str:

    unit_str = "" if p.unit_str is None else f'unit="{p.unit_str}"'
    description = "" if p.description is None else f'description="{p.description}"'

    if unit_str == "" and description == "":
        ret = f"{p.name}={p.value}"
    else:
        kwargs_str = ", ".join([unit_str, description])
        ret = f"{p.name}=ScalarParam({p.value}{kwargs_str})"

    logger.debug(f"Saving parameters {p} as {ret!r}")
    return ret


def groupby_attribute(d: Iterable[T], attr: str) -> dict[str, list[T]]:
    data = defaultdict(list)
    for di in d:
        data[getattr(di, attr)].append(di)
    return dict(data)


def print_assignment(a: atoms.Assignment) -> str:
    s = f"{a.name} = {a.expr}"
    if a.unit is not None:
        s += f" # {a.unit}"
    return s


def write_ODE_to_ode_file(ode: ODE, path: Path) -> None:
    # Just make sure it is a Path object
    path = Path(path)
    text = ""
    for component in ode.components:

        name = component.name
        logger.debug(f"Saving component {name}")

        for info, parameters in groupby_attribute(component.parameters, "info").items():
            text += start_odeblock("parameters", name=name, info=info) + "\n"
            text += ", ".join([print_ScalarParam(p) for p in parameters])
            text += "\n)\n"

        # FIXME: Add info
        for info, states in groupby_attribute(component.states, "info").items():
            text += start_odeblock("states", name=name, info=info) + "\n"
            text += ", ".join([print_ScalarParam(s) for s in states])
            text += "\n)\n"

        if name is not None:
            text += f'expressions("{name}")\n'

        text += "\n".join([print_assignment(a) for a in component.assignments])

    path.write_text(text)
