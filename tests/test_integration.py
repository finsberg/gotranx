from pathlib import Path
import pytest
import gotranx
import numpy as np
import json
from typer.testing import CliRunner

here = Path(__file__).absolute().parent
runner = CliRunner(mix_stderr=False)

rhs_program = """
from pathlib import Path
import numpy as np
import json
from scipy.integrate import solve_ivp
import {model} as model

y = model.init_state_values()
p = model.init_parameter_values()
t = np.linspace(0, {end_time}, {num_points})

res = solve_ivp(model.rhs, (0, {end_time}), y, t_eval=t, method="Radau", args=(p,))
Path("{output}").write_text(
    json.dumps(
        {{
            "t": res.t.tolist(),
            "y": res.y.tolist(),
            "status": str(res.status),
            "nfev": str(res.nfev),
            "njev": str(res.njev),
            "nlu": str(res.nlu),
            "sol": str(res.sol),
            "t_events": str(res.t_events),
            "message": str(res.message),
        }}
    )
)
"""


@pytest.mark.parametrize("path", (here / "odefiles").iterdir(), ids=lambda p: p.name)
def test_generate_python(path):
    ode = gotranx.load_ode(path)
    code = gotranx.cli.gotran2py.get_code(ode)
    ns = {}
    exec(code, ns)
    parmeters = ns["init_parameter_values"]()
    states = ns["init_state_values"]()
    rhs_zero = ns["rhs"](0, states, parmeters)
    rhs_one = ns["rhs"](1, states, parmeters)
    expected_results = json.loads(
        (here / "expected_ode_output" / path.with_suffix(".json").name).read_text()
    )
    assert np.allclose(rhs_zero, expected_results["rhs_zero"])
    assert np.allclose(rhs_one, expected_results["rhs_one"])
