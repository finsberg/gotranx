from __future__ import annotations

import os
from pathlib import Path

from .. import exceptions
from ..composition import compose, input_files
from ..ode import ODE, top_level


def top_levels(ode: ODE) -> list[str]:
    """The model's top-level components, the parts and the splits, in the order of the
    model's components"""
    return list(dict.fromkeys(top_level(c.name) for c in ode.components))


def needs(ode: ODE, component: str) -> list[str]:
    """The names the top-level ``component`` needs from the others: the missing variables
    of the model with every other top-level component subtracted, with everything nested
    under it"""
    rest = ode
    for other in top_levels(ode):
        if other != component:
            rest = rest - rest.get_component(other)
    return sorted(rest.missing_variables)


def same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    return a.exists() and b.exists() and os.path.samefile(a, b)


def main(recipe: Path, outname: Path | None = None) -> None:
    """Compose ``recipe``, write the model, and print what each part or split needs

    The model is written without guarding its singularities, like every other ``.ode``
    file: loading it guards them, as :func:`gotranx.compose` does by default.

    Raises
    ------
    exceptions.CompositionError
        If the output is one of the files the recipe reads, or the recipe cannot be
        composed. Nothing is written then.
    """
    out = Path(Path(recipe).with_suffix(".ode").name) if outname is None else Path(outname)
    for what, path in input_files(recipe):
        if same_file(out, path):
            raise exceptions.CompositionError(
                [f"Not writing {out}: it is {what}. Choose another output with -o."]
            )
    ode = compose(recipe, remove_singularities=False)
    ode.save(out)
    print(f"Wrote {out}")
    for component in top_levels(ode):
        names = needs(ode, component)
        print(f"{component} needs: {', '.join(names) if names else 'nothing'}")
    if ode.missing_variables:
        print(
            f"Warning: the model uses {', '.join(sorted(ode.missing_variables))}, which it "
            f"does not define (allow_missing). {out} can be saved but gotranx cannot load it."
        )
