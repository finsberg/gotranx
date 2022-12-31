import pytest
from gotranx import Parser
from gotranx import TreeToODE


@pytest.fixture(scope="module")
def parser() -> Parser:
    return Parser(parser="lalr", debug=True)


@pytest.fixture(scope="module")
def trans() -> TreeToODE:
    return TreeToODE()
