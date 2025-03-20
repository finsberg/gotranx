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
