from pathlib import Path

from lark import Lark


_here = Path(__file__).absolute().parent


def load_grammar() -> str:
    """Load the grammar

    Returns
    -------
    str
        The grammar
    """

    return (_here / "ode.lark").read_text()


class Parser(Lark):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(load_grammar(), *args, **kwargs)
