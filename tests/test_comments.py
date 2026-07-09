from gotranx.ode import make_ode


def test_comment_before_and_after_equation(parser, trans):
    expr = """
    states(x=1.0, y=2.0)

    # This is a comment before the equation
    dx_dt = x
    dy_dt = x - y # This is a comment after the equation
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))

    assert ode.num_states == 2
    assert ode.num_components == 1

    dx_dt = ode.get_component("").find_assignment("dx_dt")
    assert dx_dt.comment is None

    dy_dt = ode.get_component("").find_assignment("dy_dt")
    assert dy_dt.comment.text == "This is a comment after the equation"


def test_comments_inside_expressions_block(parser, trans):
    expr = """
    states("My component", x=1.0)
    expressions("My component")
    # Comment A
    dx_dt = x # Comment B
    # Comment C
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))

    assert ode.num_states == 1
    dx_dt = ode.get_component("My component").find_assignment("dx_dt")
    assert dx_dt.comment.text == "Comment B"


def test_comments_in_parameters_and_states(parser, trans):
    expr = """
    parameters(
        # Comment P1
        a=0,
        # Comment P2
        b=1 # Comment P3
        # Comment P4
    )
    states(
        # Comment S1
        x=1.0, # Comment S2
        # Comment S3
        y=2.0
        # Comment S4
    )
    dx_dt = a
    dy_dt = b
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))

    assert ode.num_parameters == 2
    assert ode.num_states == 2

    # Check that parameters were parsed correctly
    a = ode.get_component("").find_parameter("a")
    b = ode.get_component("").find_parameter("b")
    assert a.value == 0
    assert b.value == 1


def test_only_comments_in_block(parser, trans):
    expr = """
    expressions("My component")
    # Just a comment here

    parameters(
        # And here
    )

    states("My component",
        # State comments
    )
    """
    tree = parser.parse(expr)
    ode = make_ode(*trans.transform(tree))

    assert ode.num_states == 0
    assert ode.num_parameters == 0
    assert ode.num_components == 0
