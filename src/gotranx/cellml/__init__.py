from __future__ import annotations
from pathlib import Path
from typing import Any
from .cellml import CellMLParser


def cellml_to_gotran(filename: Path | str, params: dict[str, Any] | None = None) -> str:
    """Convert a cellml file to gotran code

    Parameters
    ----------
    input_filename : Path or str
        The path to the cellml file
    params : dict[str, Any], optional
        Parameters to pass to the parser, by default None

    Returns
    -------
    str
        The gotran code
    """
    parser = CellMLParser(filename, params=params)
    return parser.to_gotran()
