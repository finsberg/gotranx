from pathlib import Path
import pytest
import gotranx.myokit

here = Path(__file__).parent.absolute()


@pytest.mark.parametrize(
    "cellml_file, num_states, num_parameters",
    [("noble_1962.cellml", 4, 5), ("ToRORd_dynCl_mid.cellml", 45, 112)],
)
def test_cellml_to_gotran(cellml_file, num_states, num_parameters):
    ode = gotranx.myokit.cellml_to_gotran(
        filename=here / "cellml_files" / cellml_file,
    )
    # ode = gotranx.load.ode_from_string(text)
    assert ode.num_states == num_states
    assert ode.num_parameters == num_parameters
