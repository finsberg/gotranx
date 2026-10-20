"""Every model in `tests/odefiles`, generated for Julia and for CytoZoo.jl and run in Julia.

The other Julia tests pin what the backends print; these run it. Each model is evaluated at
its initial state by the Python backend and by the Julia one: the initial values, the
right-hand side, the monitored values and one generalized Rush-Larsen step. The CytoZoo
adapter is called through CytoZoo's interface instead. The tests need `julia` on PATH, and
the CytoZoo ones CytoZoo.jl in julia's default environment (`conftest.julia`,
`conftest.cytozoo`).

`contraction_toy.ode` has no membrane potential. It is also generated as a plain
`CytoZoo.AbstractCellModel`, checked against the Python backend, and coupled into a cardiac
model as the second node of a CytoZoo coupling, solved with OrdinaryDiffEqTsit5
(`conftest.cytozoo_solver`).
"""

from pathlib import Path

import numpy
import pytest

from gotranx.cli import gotran2cytozoo, gotran2julia, gotran2py
from gotranx.load import load_ode, ode_from_string
from gotranx.schemes import Scheme

pytestmark = pytest.mark.julia

ODEFILES = sorted((Path(__file__).parent / "odefiles").glob("*.ode"))
TOY = Path(__file__).parent / "odefiles" / "contraction_toy.ode"
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
    return python_values(ode)


def python_values(ode) -> dict:
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


# Evaluates a generated CytoZoo model, `model`, at its default initial state through CytoZoo's
# interface, and prints its state and monitor names, then the state, rhs and monitor values.
CYTOZOO_VALUES = """
u = default_initial_state(model)
du = zeros(num_states(model)); model(du, u, nothing, 0.0)
mon = zeros(num_monitors(model)); monitor_values!(mon, u, 0.0, model)
println(join(state_names(model), " ")); println(join(monitor_names(model), " "))
foreach(v -> println(join(v, " ")), (u, du, mon))
"""


def assert_cytozoo_agrees(lines, python):
    """`lines` are what CYTOZOO_VALUES printed, compared by name with the Python backend."""
    states, monitors = lines[0].split(), lines[1].split()
    assert len(states) == len(python["states"])
    by_state = [python["state_index"](name) for name in states]
    by_monitor = [python["monitor_index"](name) for name in monitors]
    assert len(set(by_monitor)) == len(python["monitors"])
    assert_close(numbers(lines[2]), python["states"][by_state])
    assert_close(numbers(lines[3]), python["rhs"][by_state])
    assert_close(numbers(lines[4]), python["monitors"][by_monitor])


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
        {CYTOZOO_VALUES}
        """
    )
    assert_cytozoo_agrees(lines, python)


def test_non_cardiac_model_in_cytozoo(cytozoo, run_julia, tmp_path):
    ode = load_ode(TOY)
    model = tmp_path / "model.jl"
    model.write_text(
        gotran2cytozoo.get_code(
            ode, model_name="Model", v_name=None, supertype="CytoZoo.AbstractCellModel"
        )
    )
    lines = run_julia(
        f"""
        using CytoZoo
        module Generated
        include("{model.as_posix()}")
        end
        model = Generated.Model()
        println(model isa CytoZoo.AbstractCellModel)
        println(model isa CytoZoo.AbstractCardiacCellModel)
        println(hasmethod(CytoZoo.transmembrane_potential_index, Tuple{{typeof(model)}}))
        {CYTOZOO_VALUES}
        """
    )
    assert lines[:3] == ["true", "false", "false"]
    assert_cytozoo_agrees(lines[3:], python_values(ode))


# A cardiac model for the toy to couple to: a resting potential, and a calcium that relaxes
# towards a source-raised level.
CARDIAC_TOY = """
parameters(v_rest=-80.0, tau_v=0.01, ca_rest=0.1, tau_ca=0.05, J_ca=2.0)
states(v=-80.0, ca=0.1)
dv_dt = -(v - v_rest) / tau_v
dca_dt = J_ca - (ca - ca_rest) / tau_ca
"""


def test_non_cardiac_model_couples_as_second_node(cytozoo_solver, run_julia, tmp_path):
    ep = tmp_path / "ep.jl"
    ep.write_text(gotran2cytozoo.get_code(ode_from_string(CARDIAC_TOY), model_name="EP"))
    mech = tmp_path / "mech.jl"
    mech.write_text(
        gotran2cytozoo.get_code(
            load_ode(TOY), model_name="Mech", v_name=None, supertype="CytoZoo.AbstractCellModel"
        )
    )
    lines = run_julia(
        f"""
        using CytoZoo, OrdinaryDiffEqTsit5, SciMLBase
        module GeneratedEP
        include("{ep.as_posix()}")
        end
        module GeneratedMech
        include("{mech.as_posix()}")
        end
        cm = couple(
            [
                Subsystem(GeneratedEP.EP(); name = :EP),
                Subsystem(GeneratedMech.Mech(); name = :Mech),
            ],
            [connect(:EP => :ca, :Mech => :Ca)],
        )
        sol = solve(ODEProblem(cm, (0.0, 0.1)), Tsit5())
        println(SciMLBase.successful_retcode(sol))
        println(all(isfinite, sol.u[end]))
        i = state_index(cm, :Mech_CaTRPN)
        println(sol.u[end][i] > 2 * sol.u[1][i])
        """
    )
    assert lines == ["true", "true", "true"]
