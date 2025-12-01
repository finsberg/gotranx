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
    pdf: bool = False,
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
    if pdf:
        import shutil

        pandoc = shutil.which("pandoc")
        if pandoc is None:
            msg = "Please install pandoc to generate PDF output. 'pip install pandoc'"
            logger.error(msg)
            raise RuntimeError(msg)

        pdf_name = out.with_suffix(".pdf")

        import subprocess as sp

        sp.run([pandoc, str(out_name), "-o", str(pdf_name)], check=True)
        logger.info(f"Wrote {pdf_name}")
