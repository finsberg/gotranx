import pytest
from gotran_parser import Parser
from gotran_parser import TreeToODE


@pytest.fixture(scope="module")
def parser() -> Parser:
    return Parser(parser="lalr", debug=True)


@pytest.fixture(scope="module")
def trans() -> TreeToODE:
    return TreeToODE()
