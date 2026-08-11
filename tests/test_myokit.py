from pathlib import Path
import math
import pytest
import gotranx.myokit

try:
    import myokit
    import myokit.formats.cellml
except ImportError:
    myokit = None

here = Path(__file__).parent.absolute()


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
@pytest.mark.parametrize(
    "cellml_file, num_states, num_parameters",
    [("noble_1962.cellml", 4, 5), ("ToRORd_dynCl_mid.cellml", 45, 112)],
)
def test_cellml_to_gotran_and_back(cellml_file, num_states, num_parameters):
    ode = gotranx.myokit.cellml_to_gotran(
        filename=here / "cellml_files" / cellml_file,
    )
    assert ode.num_states == num_states
    assert ode.num_parameters == num_parameters

    myokit_model = gotranx.myokit.gotran_to_myokit(ode)
    assert myokit_model


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
@pytest.mark.parametrize(
    "mmt_file, num_states, num_parameters",
    [("example.mmt", 8, 18)],
)
def test_mmt_to_gotran_and_back(mmt_file, num_states, num_parameters):
    ode = gotranx.myokit.mmt_to_gotran(
        filename=here / "mmt_files" / mmt_file,
    )
    assert ode.num_states == num_states
    assert ode.num_parameters == num_parameters


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
@pytest.mark.parametrize(
    "cellml_file",
    ["noble_1962.cellml", "ToRORd_dynCl_mid.cellml"],
)
def test_myokit_to_gotran_and_back(cellml_file):
    original_myokit_model = myokit.formats.cellml.CellMLImporter().model(
        here / "cellml_files" / cellml_file,
    )

    # Convert to gotran ODE
    ode = gotranx.myokit.myokit_to_gotran(original_myokit_model)
    # Convert back to myokit model
    myokit_model = gotranx.myokit.gotran_to_myokit(
        ode,
        time_component="environment",
        time_unit=gotranx.myokit.extract_unit(original_myokit_model.time().unit()),
    )

    for orig_var in original_myokit_model.variables(deep=True):
        qname = ".".join(orig_var.qname().split(".")[:-1] + [orig_var.uname()])
        if orig_var.name() in gotranx.myokit.reserved_names:
            qname = f"{qname}_"

        var = myokit_model.get(qname)
        assert var.unit() == orig_var.unit()
        assert math.isclose(var.value(), orig_var.value(), abs_tol=1e-12)


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
def test_gotran_to_myokit_cross_component_reference():
    # A state derivative in one component referencing an intermediate
    # defined in a *different* component used to raise a KeyError, because
    # the symbol substitution map in gotran_to_myokit was built from plain
    # sp.Symbol(name) objects while gotranx atoms use sp.Symbol(name,
    # real=True, ...) - the two are never equal, so xreplace silently did
    # nothing for any cross-component reference.
    ode = gotranx.load.ode_from_string(
        """
        states("membrane", V=ScalarParam(-87, unit="mV", description=""))

        parameters("leak",
        E_L=ScalarParam(-60.0, unit="mV", description=""),
        g_L=ScalarParam(75.0, unit="uS", description="")
        )

        parameters("membrane",
        Cm=ScalarParam(12.0, unit="uF", description="")
        )

        expressions("leak")
        i_Leak = g_L*(-E_L + V) # nA

        expressions("membrane")
        dV_dt = -i_Leak/Cm # mV
        """
    )
    myokit_model = gotranx.myokit.gotran_to_myokit(ode)
    v = myokit_model.get("membrane.V")
    # The right-hand side must reference the *qualified* leak.i_Leak variable,
    # not the bare (and therefore unresolved) name "i_Leak".
    assert "leak.i_Leak" in v.rhs().code()


@pytest.mark.skipif(myokit is None, reason="myokit not installed")
@pytest.mark.parametrize(
    "component_name",
    ["", "My component", "environment"],
    ids=["unnamed", "with-space", "reserved"],
)
def test_gotran_to_myokit_sanitizes_component_names(component_name):
    # Component names in gotranx are free-form strings: they default to the
    # empty string when not given explicitly, and may contain spaces or other
    # characters that are not valid myokit/CellML identifiers. Both used to
    # raise myokit.InvalidNameError.
    if component_name:
        text = f'states("{component_name}", x=1.0)\nexpressions("{component_name}")\ndx_dt = -x\n'
    else:
        text = "states(x=1.0)\ndx_dt = -x\n"
    ode = gotranx.load.ode_from_string(text)

    myokit_model = gotranx.myokit.gotran_to_myokit(ode)
    myokit_model.validate()
