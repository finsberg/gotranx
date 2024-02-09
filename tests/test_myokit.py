from pathlib import Path
import pytest
import gotranx.myokit

here = Path(__file__).parent.absolute()


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


@pytest.mark.parametrize(
    "mmt_file, num_states, num_parameters",
    [("example.mmt", 8, 18)],
)
def test_mmt_to_gotran(mmt_file, num_states, num_parameters):
    ode = gotranx.myokit.mmt_to_gotran(
        filename=here / "mmt_files" / mmt_file,
    )
    assert ode.num_states == num_states
    assert ode.num_parameters == num_parameters
