from __future__ import annotations
from pathlib import Path
import logging
import structlog

from ..myokit import gotran_to_cellml
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
    assert fname.suffix in ".ode", f"File {fname} must be an ode file"
    assert fname.exists(), f"File {fname} does not exist"
    ode = load_ode(fname)
    logger.info(f"Converting {fname} to CellML")
    cellml_file = fname.with_suffix(".cellml") if outname is None else Path(outname)
    gotran_to_cellml(ode, filename=cellml_file)
    logger.info(f"Wrote {cellml_file}")
