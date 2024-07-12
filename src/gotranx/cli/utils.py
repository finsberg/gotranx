from __future__ import annotations
from typing import Any
from ..codegen import CodeGenerator
from ..schemes import Scheme, get_scheme


def add_schemes(
    codegen: CodeGenerator,
    scheme: list[Scheme] | None = None,
    delta: float = 1e-8,
) -> list[str]:
    comp = []
    if scheme is not None:
        for s in scheme:
            kwargs: dict[str, Any] = {}
            if "rush_larsen" in s.value:
                kwargs["delta"] = delta

            comp.append(codegen.scheme(get_scheme(s.value), **kwargs))
    return comp
