from textwrap import dedent

import pytest
from gotranx.codegen import MTKCodeGenerator
from gotranx.ode import make_ode


def _lorentz_expr() -> str:
    return dedent(
        """
        parameters(a=0)
        parameters("My component",
        sigma=ScalarParam(12.0, description="Some description"),
        rho=21.0,
        beta=2.4
        )
        states("My component", x=1.0, y=2.0,z=3.05)

        expressions("My component")
        rhoz = rho - z
        dy_dt = x*rhoz - y # millivolt
        dx_dt = sigma*(-x + y)
        betaz = beta*z
        dz_dt = -betaz + x*y
        """,
    )


def test_mtk_codegen(parser, trans):
    tree = parser.parse(_lorentz_expr())
    ode = make_ode(*trans.transform(tree), name="lorentz")
    codegen = MTKCodeGenerator(ode)
    code = codegen.generate()

    assert "@parameters t a beta rho sigma" in code
    assert "@variables x(t) z(t) y(t)" in code
    assert "betaz ~ beta .* z" in code
    assert "D(x) ~ sigma .* (-x + y)" in code
    assert "lorentz = ODESystem" in code
    assert "x => 1.0" in code and "sigma => 12.0" in code


@pytest.fixture(scope="module")
def odefile(tmp_path_factory):
    fname = tmp_path_factory.mktemp("data") / "lorentz.ode"
    fname.write_text(_lorentz_expr())
    yield fname
    fname.unlink()
