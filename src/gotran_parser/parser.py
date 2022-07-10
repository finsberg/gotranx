from pathlib import Path

from lark import Lark


_here = Path(__file__).absolute().parent


def load_grammar() -> str:
    with open(_here / "ode.lark", "r") as f:
        text = f.read()
    return text


class Parser(Lark):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(load_grammar(), *args, **kwargs)
