from __future__ import annotations

from pathlib import Path

from structlog import get_logger

from .codegen.ode import GotranODECodePrinter
from .ode import ODE

logger = get_logger()


def write_ODE_to_ode_file(ode: ODE, path: Path) -> None:
    # Just make sure it is a Path object
    printer = GotranODECodePrinter(ode)
    path = Path(path)
    text = []
    text.append(printer.print_comments())
    text.append(printer.print_states())
    text.append(printer.print_parameters())
    text.append(printer.print_assignments())
    path.write_text("".join(text))
    logger.info(f"Wrote {path}")
