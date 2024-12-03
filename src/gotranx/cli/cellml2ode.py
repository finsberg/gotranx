from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..myokit import cellml_to_gotran

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
    assert fname.suffix in (".cellml", ".xml"), f"File {fname} must be a cellml or xml file"
    assert fname.exists(), f"File {fname} does not exist"
    logger.info(f"Converting {fname} to gotran ode file")
    ode = cellml_to_gotran(fname)
    out = fname.with_suffix(".ode") if outname is None else Path(outname)
    ode.save(out)
    logger.info(f"Wrote {out}")
