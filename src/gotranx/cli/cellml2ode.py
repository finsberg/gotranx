from __future__ import annotations
from pathlib import Path
from structlog import get_logger

from ..cellml import cellml_to_gotran

logger = get_logger()


def main(
    fname: Path,
    outname: str | None = None,
) -> None:
    assert fname.suffix == ".cellml", f"File {fname} must be a cellml file"
    assert fname.exists(), f"File {fname} does not exist"
    logger.info(f"Converting {fname} to gotran ode file")
    code = cellml_to_gotran(fname)
    out = fname.with_suffix(".ode") if outname is None else Path(outname)
    out.write_text(code)
    logger.info(f"Wrote {out}")
