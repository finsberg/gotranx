import gotranx


def test_component_to_ode(trans, parser):
    expr = """
    parameters("Main component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0
    )

    parameters("Z component",
    beta=2.4
    )

    states("Main component", x=1.0, y=2.0)

    states("Z component", z=3.05)

    expressions("Main component")
    rhoz = rho - z
    dy_dt = x*rhoz - y # millivolt
    dx_dt = sigma*(-x + y)

    expressions("Z component")
    betaz = beta*z
    dz_dt = -betaz + x*y
    """

    tree = parser.parse(expr)
    main_ode = gotranx.ode.make_ode(*trans.transform(tree), name="lorentz")
    z_comp = main_ode.get_component("Z component")
    z_ode = z_comp.to_ode()
    assert z_ode.missing_variables == {"x", "y"}
    assert z_ode.num_states == 1
    assert z_ode.num_parameters == 1
    assert z_ode.parameters[0].name == "beta"
    assert z_ode.states[0].name == "z"
    assert z_ode.name == "Z component"

    remaining_ode = main_ode - z_comp
    assert remaining_ode.missing_variables == {"z"}
    assert remaining_ode.num_states == 2
    assert remaining_ode.num_parameters == 2
    assert remaining_ode.parameters[0].name == "rho"
    assert remaining_ode.parameters[1].name == "sigma"
    assert remaining_ode.states[0].name == "x"
    assert remaining_ode.states[1].name == "y"
    assert remaining_ode.name == "lorentz - Z component"
