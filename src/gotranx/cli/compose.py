from __future__ import annotations

import os
from pathlib import Path

from .. import exceptions
from ..composition import compose, input_files
from ..ode import ODE


def needs(ode: ODE, component: str) -> list[str]:
    """The names ``component`` needs from the other components: the missing variables
    of the model with every other component subtracted"""
    rest = ode
    for other in ode.components:
        if other.name != component:
            rest = rest - rest.get_component(other.name)
    return sorted(rest.missing_variables)


def same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    return a.exists() and b.exists() and os.path.samefile(a, b)


def main(recipe: Path, outname: Path | None = None) -> None:
    """Compose ``recipe``, write the model, and print what each component needs

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
    for component in ode.components:
        names = needs(ode, component.name)
        print(f"{component.name} needs: {', '.join(names) if names else 'nothing'}")
    if ode.missing_variables:
        print(
            f"Warning: the model uses {', '.join(sorted(ode.missing_variables))}, which it "
            f"does not define (allow_missing). {out} can be saved but gotranx cannot load it."
        )
