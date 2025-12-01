from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..codegen.markdown import MarkdownGenerator
from ..load import load_ode

logger = structlog.get_logger()


def main(
    fname: Path,
    outname: str | None = None,
    verbose: bool = False,
) -> None:
    loglevel = logging.DEBUG if verbose else logging.INFO
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(loglevel),
    )

    ode = load_ode(fname)
    generator = MarkdownGenerator(ode)
    code = generator.generate()

    out = fname if outname is None else Path(outname)
    out_name = out.with_suffix(".md")
    out_name.write_text(code)
    logger.info(f"Wrote {out_name}")
