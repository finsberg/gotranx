from __future__ import annotations

from pathlib import Path

from ..composition import compose
from ..ode import ODE


def needs(ode: ODE, component: str) -> list[str]:
    """The names ``component`` needs from the other components: the missing variables
    of the model with every other component subtracted"""
    rest = ode
    for other in ode.components:
        if other.name != component:
            rest = rest - rest.get_component(other.name)
    return sorted(rest.missing_variables)


def main(recipe: Path, outname: Path | None = None) -> None:
    """Compose ``recipe``, write the model, and print what each component needs

    Raises
    ------
    exceptions.CompositionError
        If the recipe cannot be composed. Nothing is written then.
    """
    ode = compose(recipe)
    out = Path(Path(recipe).with_suffix(".ode").name) if outname is None else Path(outname)
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
