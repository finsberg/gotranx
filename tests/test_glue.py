import lark
import pytest
import sympy as sp

from gotranx import atoms
from gotranx.exceptions import GotranxError
from gotranx.expressions import build_expression
from gotranx.glue import load_glue, parse_glue, unqualify
from gotranx.load import ode_from_string
from gotranx.parser import Parser

# The spec's ToR-ORd + Land glue file, verbatim.
TORORD_GLUE = """\
# How ep (ToR-ORd) and land (Land 2017) are coupled.
# In a block for part X, bare names are X's; another part's are written part.name.

parameters("land",
lmbda = 1.0,                              # new: fibre stretch, set by simcardemsx
dLambda = ScalarParam(0.0, unit="1/ms")   # new: its rate
)

expressions("land")
replace Ca = 1000.0*ep.cai                # parameter -> input: mM to uM
replace SL = lmbda*SL0
replace dSL = 1000.0*dLambda*SL0          # 1/ms to um/s, Land's own unit
J_TRPN = ep.trpnmax*dCaTRPN_dt            # new: troponin flux, mM/ms

expressions("ep")
replace Bcai = 1.0/(1.0 + cmdnmax*kmcmdn/(cai + kmcmdn)**2)   # troponin is Land's state now
dcai_dt += -Bcai*land.J_TRPN                                   # Margara 2021
"""


def test_parses_the_torord_land_glue():
    s = parse_glue(TORORD_GLUE)
    assert [(x.kind, x.block, x.component, x.name) for x in s] == [
        ("new", "parameters", "land", "lmbda"),
        ("new", "parameters", "land", "dLambda"),
        ("replace", "expressions", "land", "Ca"),
        ("replace", "expressions", "land", "SL"),
        ("replace", "expressions", "land", "dSL"),
        ("new", "expressions", "land", "J_TRPN"),
        ("replace", "expressions", "ep", "Bcai"),
        ("add", "expressions", "ep", "dcai_dt"),
    ]
    assert ("ep", "cai") in s[2].references and ("land", "J_TRPN") in s[7].references


def test_references_and_atoms():
    s = parse_glue(TORORD_GLUE)
    assert s[0].references == frozenset()
    assert s[3].references == {(None, "lmbda"), (None, "SL0")}
    assert s[5].references == {("ep", "trpnmax"), (None, "dCaTRPN_dt")}
    assert s[6].references == {(None, "cmdnmax"), (None, "kmcmdn"), (None, "cai")}
    assert s[7].references == {(None, "Bcai"), ("land", "J_TRPN")}

    assert isinstance(s[1].atom, atoms.Parameter)
    assert s[1].atom.value == 0.0 and s[1].atom.unit_str == "1/ms"
    assert s[1].atom.components == ("land",)
    for x in s[2:]:
        assert isinstance(x.atom, atoms.Assignment)
        assert x.atom.name == x.name and x.atom.components == (x.component,)
    # The tree keeps the qualified name; unqualify gives a plain variable
    sym = {"cai": sp.Symbol("cai")}
    assert build_expression(unqualify(s[2].atom.value.tree), symbols=sym) == 1000.0 * sym["cai"]


def test_replace_in_parameters_block():
    (s,) = parse_glue('parameters("land",\nreplace lmbda = ScalarParam(1.0, unit="1")\n)')
    assert (s.kind, s.block, s.component, s.name) == ("replace", "parameters", "land", "lmbda")
    assert s.atom.value == 1.0 and s.atom.unit_str == "1" and s.line == 2

    (s,) = parse_glue('parameters("land", replace lmbda = 2.0)')
    assert (s.kind, s.name, s.atom.value) == ("replace", "lmbda", 2.0)


def test_line_numbers():
    lines = TORORD_GLUE.splitlines()
    s = parse_glue(TORORD_GLUE)
    assert [x.line for x in s] == [5, 6, 10, 11, 12, 13, 16, 17]
    for x in s:
        line = lines[x.line - 1]
        assert line.startswith("replace " if x.kind == "replace" else x.name)
        assert x.name in line.split("=")[0]


def test_load_glue(tmp_path):
    path = tmp_path / "torord_land.glue.ode"
    path.write_text(TORORD_GLUE)
    assert load_glue(path) == parse_glue(TORORD_GLUE)
    assert load_glue(str(path)) == parse_glue(TORORD_GLUE)


@pytest.mark.parametrize(
    "text",
    [
        'expressions("a")\nreplace x = 1',
        'expressions("a")\ndy_dt += 1',
        'expressions("a")\nx = b.c',
        'parameters("a",\nreplace x = 1\n)',
        'parameters("a",\nreplace x = ScalarParam(1.0)\n)',
        "x = 2\ndy_dt += x",
    ],
)
def test_plain_ode_refuses_glue_constructs(text):
    with pytest.raises(GotranxError, match="only allowed in glue files") as e:
        ode_from_string(text)
    assert str(e.value) == "`replace`, `+=` and `part.name` are only allowed in glue files"


def test_plain_ode_refuses_glue_constructs_after_parsing(parser, trans):
    # The tests' parser builds the tree, and the transformer runs afterwards
    with pytest.raises(GotranxError, match="only allowed in glue files"):
        trans.transform(parser.parse('expressions("a")\nx = 2*b.c'))


def test_replace_is_not_a_reserved_word():
    # Review Focus 1
    ode = ode_from_string('parameters("a", replace=1.0)\nexpressions("a")\nx = 2*replace')
    assert ode["x"].expr == 2 * ode["replace"].symbol
    assert ode["x"].expr == 2 * sp.Symbol("replace", real=True, finite=True)

    ode = ode_from_string('parameters("a", y=1.0)\nexpressions("a")\nreplace = 2*y\nx = replace')
    assert ode["x"].expr == ode["replace"].symbol


def test_replace_is_not_a_reserved_word_in_glue():
    s = parse_glue('parameters("a", replace=1.0)\nexpressions("a")\nx = 2*replace')
    assert [(x.kind, x.name) for x in s] == [("new", "replace"), ("new", "x")]
    assert s[1].references == {(None, "replace")}


def test_unknown_statement_word():
    with pytest.raises(GotranxError, match="unknown statement foo"):
        parse_glue('expressions("a")\nfoo x = 1')
    with pytest.raises(GotranxError, match="unknown statement foo"):
        parse_glue('parameters("a",\nfoo x = 1\n)')


def test_unknown_statement_word_in_a_plain_ode():
    with pytest.raises(GotranxError, match="unknown statement foo"):
        ode_from_string('expressions("a")\nfoo x = 1')
    with pytest.raises(GotranxError, match="unknown statement foo"):
        ode_from_string('parameters("a",\nfoo x = 1\n)')


def test_glue_cannot_declare_states():
    with pytest.raises(GotranxError, match="glue files cannot declare states"):
        parse_glue('states("a", y=1.0)\nexpressions("a")\nx = y')


@pytest.mark.parametrize(
    "text",
    [
        "x = 1",
        'expressions("a", "b")\nx = 1',
        "parameters(x=1.0)",
        'parameters("a", "b", x=1.0)',
    ],
)
def test_glue_blocks_name_exactly_one_component(text):
    with pytest.raises(GotranxError, match="exactly one component"):
        parse_glue(text)


def test_unqualify():
    tree = Parser(parser="lalr", propagate_positions=True).parse('expressions("a")\nx = 2*ep.cai')
    (expr,) = [t for t in tree.iter_subtrees() if t.data == "term"]
    assert any(t.data == "qualified_variable" for t in expr.iter_subtrees())

    plain = unqualify(expr)
    assert not any(t.data == "qualified_variable" for t in plain.iter_subtrees())
    assert build_expression(plain, symbols={"cai": sp.Symbol("cai")}) == 2 * sp.Symbol("cai")
    assert atoms.Expression(tree=plain).dependencies == {"cai"}
    # The original tree is left alone
    assert any(t.data == "qualified_variable" for t in expr.iter_subtrees())


def test_unqualify_keeps_line_numbers():
    s = parse_glue('expressions("a")\n\nx = 2*ep.cai')
    (var,) = [t for t in unqualify(s[0].atom.value.tree).iter_subtrees() if t.data == "variable"]
    assert var.meta.line == 3
    with pytest.raises(GotranxError, match="line 3"):
        build_expression(unqualify(s[0].atom.value.tree), symbols={})


def test_unqualify_a_bare_variable():
    tree = lark.Tree(
        "qualified_variable", [lark.Token("VARIABLE", "ep"), lark.Token("VARIABLE", "v")]
    )
    assert unqualify(tree) == lark.Tree("variable", [lark.Token("VARIABLE", "v")])


def test_glue_errors_name_the_line():
    with pytest.raises(GotranxError, match="line 3: unknown statement foo"):
        parse_glue('expressions("a")\nx = 1\nfoo y = 2')
    with pytest.raises(GotranxError, match="line 2: glue files cannot declare states"):
        parse_glue('\nstates("a", y=1.0)')


def test_trailing_comment_with_an_apostrophe():
    # pint raises tokenize.TokenError on "Land's", and the spec's glue file has two
    # such comments. They are comments, in a plain .ode file too.
    ode = ode_from_string('parameters("a", y=1.0)\nexpressions("a")\nx = y # Land\'s own unit')
    assert ode["x"].comment == atoms.Comment("Land's own unit")
    assert parse_glue(TORORD_GLUE)[4].atom.comment == atoms.Comment("1/ms to um/s, Land's own unit")
