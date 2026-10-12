"""Every model in `tests/odefiles`, generated for Julia and for CytoZoo.jl and run in Julia.

The other Julia tests pin what the backends print; these run it. Each model is evaluated at
its initial state by the Python backend and by the Julia one: the initial values, the
right-hand side, the monitored values and one generalized Rush-Larsen step. The CytoZoo
adapter is called through CytoZoo's interface instead. The tests need `julia` on PATH, and
the CytoZoo ones CytoZoo.jl in julia's default environment (`conftest.julia`,
`conftest.cytozoo`).
"""

from pathlib import Path

import numpy
import pytest

from gotranx.cli import gotran2cytozoo, gotran2julia, gotran2py
from gotranx.load import load_ode
from gotranx.schemes import Scheme

pytestmark = pytest.mark.julia

ODEFILES = sorted((Path(__file__).parent / "odefiles").glob("*.ode"))
SCHEME = [Scheme.generalized_rush_larsen]
DT = 0.01


def assert_close(actual, desired):
    # Julia's math functions are not numpy's, and the right-hand side of a cell model at rest
    # is a sum of currents that nearly cancel, so the agreement is to round-off of the terms,
    # not of the sum.
    numpy.testing.assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


def numbers(line: str) -> numpy.ndarray:
    return numpy.array([float(x) for x in line.split()])


@pytest.fixture(scope="module", params=ODEFILES, ids=lambda path: path.stem)
def odefile(request) -> Path:
    return request.param


@pytest.fixture(scope="module")
def ode(odefile):
    return load_ode(odefile)


@pytest.fixture(scope="module")
def python(ode) -> dict:
    """The Python backend's values at the model's initial state."""
    namespace: dict = {}
    exec(gotran2py.get_code(ode, scheme=SCHEME), namespace)
    states = namespace["init_state_values"]()
    parameters = namespace["init_parameter_values"]()
    return {
        "states": states,
        "parameters": parameters,
        "rhs": namespace["rhs"](0.0, states, parameters),
        "monitors": namespace["monitor_values"](0.0, states, parameters),
        "step": namespace["generalized_rush_larsen"](states, 0.0, DT, parameters),
        "state_index": namespace["state_index"],
        "monitor_index": namespace["monitor_index"],
    }


@pytest.mark.parametrize("type_stable", [False, True], ids=["plain", "type_stable"])
def test_ode2julia_agrees_with_python(ode, python, run_julia, tmp_path, type_stable):
    model = tmp_path / "model.jl"
    model.write_text(gotran2julia.get_code(ode, scheme=SCHEME, type_stable=type_stable))
    lines = run_julia(
        f"""
        include("{model.as_posix()}")
        s = zeros(NUM_STATES); init_state_values!(s)
        p = zeros(NUM_PARAMS); init_parameter_values!(p)
        out = zeros(NUM_STATES); rhs(0.0, s, p, out)
        mon = zeros(NUM_MONITORED); monitor_values(0.0, s, p, mon)
        st = zeros(NUM_STATES); generalized_rush_larsen(s, 0.0, {DT}, p, st)
        foreach(v -> println(join(v, " ")), (s, p, out, mon, st))
        """
    )
    keys = ("states", "parameters", "rhs", "monitors", "step")
    assert len(lines) == len(keys)
    for key, line in zip(keys, lines):
        assert_close(numbers(line), python[key])


def v_name(ode) -> str:
    names = [state.name for state in ode.states]
    return next((name for name in ("v", "V") if name in names), names[0])


def test_ode2cytozoo_agrees_with_python(ode, python, cytozoo, run_julia, tmp_path):
    model = tmp_path / "model.jl"
    model.write_text(gotran2cytozoo.get_code(ode, model_name="Model", v_name=v_name(ode)))
    lines = run_julia(
        f"""
        using CytoZoo
        module Generated
        include("{model.as_posix()}")
        end
        model = Generated.Model()
        u = default_initial_state(model)
        du = zeros(num_states(model)); model(du, u, nothing, 0.0)
        mon = zeros(num_monitors(model)); monitor_values!(mon, u, 0.0, model)
        println(join(state_names(model), " ")); println(join(monitor_names(model), " "))
        foreach(v -> println(join(v, " ")), (u, du, mon))
        """
    )
    states, monitors = lines[0].split(), lines[1].split()
    assert len(states) == len(python["states"])
    by_state = [python["state_index"](name) for name in states]
    by_monitor = [python["monitor_index"](name) for name in monitors]
    assert len(set(by_monitor)) == len(python["monitors"])
    assert_close(numbers(lines[2]), python["states"][by_state])
    assert_close(numbers(lines[3]), python["rhs"][by_state])
    assert_close(numbers(lines[4]), python["monitors"][by_monitor])
