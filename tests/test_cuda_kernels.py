"""The CUDA kernels of every model in `tests/odefiles`, run on the CPU.

`cuda_emulation.h` lets a C++ compiler compile the generated kernels and run them one thread
at a time, so their indexing and numerics can be checked without a GPU: each model is run on
several cells with different states, stored state-major (`(num_states, n_cells)`), and
compared against the Python backend on the same array. Compiling with nvcc is
`test_nvcc.py`. They need a C++ compiler (`conftest.cxx_compiler`).
"""

import ctypes
import math
from pathlib import Path

import numpy
import pytest

from gotranx.cli import gotran2cuda, gotran2py
from gotranx.codegen import CFormat
from gotranx.load import load_ode, ode_from_string
from gotranx.schemes import Scheme

HERE = Path(__file__).parent
# ohara_rudy_cipa_v1_2017.ode is a git-ignored local file that does not load
# (DuplicateSymbolError), so it only exists in some checkouts. Skip it until it is fixed.
ODEFILES = sorted(
    path for path in (HERE / "odefiles").glob("*.ode") if path.name != "ohara_rudy_cipa_v1_2017.ode"
)
CARDIAC_MODEL = HERE / "odefiles" / "ToRORd_dyn_chloride.ode"
SCHEMES = [Scheme.generalized_rush_larsen, Scheme.forward_explicit_euler]
N_CELLS = 5
DT = 0.01
T = 0.0

# The kernels' leading arguments (as in C), by kind, and how to pass them on
SIGNATURES = {
    "init": ("double* s", "s"),
    "rhs": ("double t, const double* s, const double* p, double* v", "t, s, p, v"),
    "scheme": (
        "const double* s, double t, double dt, const double* p, double* v",
        "s, t, dt, p, v",
    ),
}
ARGTYPES: dict[str, list] = {
    "init": [ctypes.c_void_p],
    "rhs": [ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p],
    "scheme": [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_void_p, ctypes.c_void_p],
}


def assert_close(actual, desired):
    numpy.testing.assert_allclose(actual, desired, rtol=1e-12, atol=1e-12)


class Kernels:
    """The kernels of a generated header, compiled for the CPU.

    `run(name, *args, n_cells=..., blocks=..., threads=...)` launches kernel `name` on
    `blocks` blocks of `threads` threads (by default one block per cell); `args` are the
    kernel's arguments before `n_cells`, numpy arrays passed as pointers.
    """

    def __init__(
        self, code: str, namespace: str, kernels: dict, missing: bool, directory: Path, compiler
    ):
        lines = ['#include "cuda_emulation.h"', '#include "model.cuh"', 'extern "C" {']
        lines.append(
            "void host_init_parameter_values(double* p) "
            f"{{ {namespace}::init_parameter_values(p); }}"
        )
        for name, kind in kernels.items():
            params, args = SIGNATURES[kind]
            if missing and kind != "init":
                params += ", const double* m"
                args += ", m"
            lines.append(
                f"void run_{name}({params}, long long n_cells, long long blocks, unsigned threads) "
                "{ gotranx_launch(blocks, threads, [&] "
                f"{{ {namespace}::{name}({args}, n_cells); }}); }}"
            )
        lines.append("}")
        (directory / "model.cuh").write_text(code)
        source = directory / "model.cpp"
        source.write_text("\n".join(lines) + "\n")
        # Every model gets its own directory, and so its own library: ctypes would return an
        # already loaded library for a path it has loaded before
        # The kernels are `static __global__`; with __global__ defined away, the ones this
        # test does not wrap are unused static functions, which only the emulation sees
        flags = ("-I", str(HERE), "-Wno-unused-function")
        self.lib = ctypes.CDLL(str(compiler.with_flags(*flags).shared_library(source)))
        for name, kind in kernels.items():
            argtypes = list(ARGTYPES[kind])
            if missing and kind != "init":
                argtypes.append(ctypes.c_void_p)
            getattr(self.lib, f"run_{name}").argtypes = argtypes + [
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_uint,
            ]
        self.lib.host_init_parameter_values.argtypes = [ctypes.c_void_p]

    def run(self, name, *args, n_cells, blocks=None, threads=1):
        if blocks is None:
            blocks = math.ceil(n_cells / threads)
        args = [a.ctypes.data if isinstance(a, numpy.ndarray) else a for a in args]
        getattr(self.lib, f"run_{name}")(*args, n_cells, blocks, threads)


def kinds(schemes) -> dict:
    return {
        "init_state_values": "init",
        "rhs": "rhs",
        "monitor_values": "rhs",
        **{s.value: "scheme" for s in schemes},
    }


def perturbed(states: numpy.ndarray, n_cells: int = N_CELLS) -> numpy.ndarray:
    """`(num_states, n_cells)`: each cell's states differ by about 1 %."""
    rng = numpy.random.default_rng(0)
    return numpy.ascontiguousarray(numpy.outer(states, 1 + 0.01 * rng.standard_normal(n_cells)))


def python_namespace(ode, **kwargs) -> dict:
    namespace: dict = {}
    exec(gotran2py.get_code(ode, **kwargs), namespace)
    return namespace


@pytest.fixture(scope="module", params=ODEFILES, ids=lambda path: path.stem)
def ode(request):
    return load_ode(request.param)


@pytest.fixture(scope="module")
def kernels(ode, cxx_compiler, tmp_path_factory) -> Kernels:
    code = gotran2cuda.get_code(ode, scheme=SCHEMES, format=CFormat.none)
    return Kernels(
        code, ode.name, kinds(SCHEMES), False, tmp_path_factory.mktemp("cuda"), cxx_compiler
    )


@pytest.fixture(scope="module")
def python(ode) -> dict:
    return python_namespace(ode, scheme=SCHEMES)


@pytest.fixture(scope="module")
def cells(python) -> numpy.ndarray:
    return perturbed(python["init_state_values"]())


def test_cuda_init_state_values_fills_every_cell(kernels, python):
    states = numpy.zeros((len(python["init_state_values"]()), N_CELLS))
    kernels.run("init_state_values", states, n_cells=N_CELLS)
    assert_close(states, numpy.outer(python["init_state_values"](), numpy.ones(N_CELLS)))


def test_cuda_init_parameter_values_is_a_host_function(kernels, python):
    parameters = numpy.zeros_like(python["init_parameter_values"]())
    kernels.lib.host_init_parameter_values(parameters.ctypes.data)
    assert_close(parameters, python["init_parameter_values"]())


def test_cuda_rhs_and_monitors_agree_with_python(kernels, python, cells):
    parameters = python["init_parameter_values"]()
    out = numpy.zeros_like(cells)
    kernels.run("rhs", T, cells, parameters, out, n_cells=N_CELLS)
    assert_close(out, python["rhs"](T, cells, parameters))

    expected = python["monitor_values"](T, cells, parameters)
    out = numpy.zeros_like(expected)
    kernels.run("monitor_values", T, cells, parameters, out, n_cells=N_CELLS)
    assert_close(out, expected)


@pytest.mark.parametrize("scheme", [s.value for s in SCHEMES])
def test_cuda_schemes_agree_with_python(kernels, python, cells, scheme):
    parameters = python["init_parameter_values"]()
    out = numpy.zeros_like(cells)
    kernels.run(scheme, cells, T, DT, parameters, out, n_cells=N_CELLS)
    assert_close(out, python[scheme](cells, T, DT, parameters))


def test_cuda_step_in_place_equals_out_of_place(kernels, python, cells):
    parameters = python["init_parameter_values"]()
    out = numpy.zeros_like(cells)
    kernels.run("generalized_rush_larsen", cells, T, DT, parameters, out, n_cells=N_CELLS)
    in_place = cells.copy()
    kernels.run("generalized_rush_larsen", in_place, T, DT, parameters, in_place, n_cells=N_CELLS)
    assert numpy.array_equal(in_place, out)


def test_cuda_trailing_threads_do_not_write(kernels, python, cells):
    # 2 blocks of 4 threads for 5 cells: threads 5-7 must return without writing
    parameters = python["init_parameter_values"]()
    guards = 3
    size = cells.size
    buffer = numpy.full(size + guards, numpy.nan)
    kernels.run("rhs", T, cells, parameters, buffer, n_cells=N_CELLS, blocks=2, threads=4)
    assert numpy.isnan(buffer[size:]).all()
    expected = numpy.zeros_like(cells)
    kernels.run("rhs", T, cells, parameters, expected, n_cells=N_CELLS)
    assert numpy.array_equal(buffer[:size].reshape(cells.shape), expected)


# --- Other code paths ---------------------------------------------------------------------------


def test_cuda_missing_variables_per_cell_agree_with_python(cxx_compiler, tmp_path):
    from test_missing_variables import TEXT

    full = ode_from_string(TEXT)
    ode = full - full.get_component("inputs")
    schemes = [Scheme.generalized_rush_larsen]
    # Subtracting a component names the ODE "ode - inputs", which is no namespace
    code = gotran2cuda.get_code(ode, scheme=schemes, format=CFormat.none, namespace="split")
    kernels = Kernels(code, "split", kinds(schemes), True, tmp_path, cxx_compiler)
    python = python_namespace(ode, scheme=schemes)

    cells = numpy.array([[1.0, 1.1, 0.9], [0.25, 0.3, 0.2]])
    missing = numpy.array([[2.5, 2.0, 3.0], [-1.5, -1.0, -2.0]])  # u_in, w_in per cell
    parameters = python["init_parameter_values"]()

    out = numpy.zeros_like(cells)
    kernels.run("rhs", T, cells, parameters, out, missing, n_cells=3)
    assert_close(out, python["rhs"](T, cells, parameters, missing))

    out = numpy.zeros_like(cells)
    kernels.run("generalized_rush_larsen", cells, T, 0.1, parameters, out, missing, n_cells=3)
    assert_close(out, python["generalized_rush_larsen"](cells, T, 0.1, parameters, missing))


@pytest.fixture(scope="module")
def tor_ord():
    return load_ode(CARDIAC_MODEL)


def test_cuda_compiles_with_remove_unused(tor_ord, cxx_compiler, tmp_path):
    code = gotran2cuda.get_code(
        tor_ord, scheme=list(Scheme), remove_unused=True, format=CFormat.none
    )
    Kernels(code, tor_ord.name, kinds([]), False, tmp_path, cxx_compiler)


def test_cuda_compiles_hybrid_rush_larsen_with_stiff_states(tor_ord, cxx_compiler, tmp_path):
    code = gotran2cuda.get_code(
        tor_ord,
        scheme=[Scheme.hybrid_rush_larsen],
        stiff_states=["v", "cai"],
        format=CFormat.none,
    )
    Kernels(code, tor_ord.name, kinds([Scheme.hybrid_rush_larsen]), False, tmp_path, cxx_compiler)


def test_cuda_missing_values_agree_with_python(tor_ord, cxx_compiler, tmp_path):
    values = {"INa": 0, "cai": 1}
    code = gotran2cuda.get_code(tor_ord, missing_values=values, format=CFormat.none)
    assert "__global__ void missing_values(" in code
    kernels = Kernels(code, tor_ord.name, {"missing_values": "rhs"}, False, tmp_path, cxx_compiler)
    python = python_namespace(tor_ord, missing_values=values)
    cells = perturbed(python["init_state_values"]())
    parameters = python["init_parameter_values"]()
    out = numpy.zeros((len(values), N_CELLS))
    kernels.run("missing_values", T, cells, parameters, out, n_cells=N_CELLS)
    assert_close(out, python["missing_values"](T, cells, parameters))


# --- The cable demo (examples/cuda-cable) -------------------------------------------------------

CABLE_DEMO = HERE / ".." / "examples" / "cuda-cable"


def test_cuda_cable_propagates(cxx_compiler, tmp_path):
    """The demo's kernels and constants (`cable.cuh`), with the host loop of `main.cu` on the
    CPU: a short cable, stimulated at one end, must conduct a wave to the other."""
    ode = load_ode(CABLE_DEMO / "ToRORd_dyn_chloride.ode")
    code = gotran2cuda.get_code(
        ode, scheme=[Scheme.generalized_rush_larsen], format=CFormat.none, namespace="tor_ord"
    )
    (tmp_path / "tor_ord.cuh").write_text(code)
    source = tmp_path / "cable.cpp"
    source.write_text(
        """
#include "cuda_emulation.h"
#include "tor_ord.cuh"
#include "cable.cuh"
#include <vector>

// The time each cell's potential first exceeds 0 mV, or -1
extern "C" void simulate(long long n_cells, double t_end, double* activation)
{
    std::vector<double> parameters(tor_ord::NUM_PARAMS);
    tor_ord::init_parameter_values(parameters.data());
    parameters[tor_ord::parameter_index("i_Stim_Amplitude")] = 0.0;
    const int v = tor_ord::state_index("v");
    std::vector<double> states(tor_ord::NUM_STATES * n_cells), v_new(n_cells);
    gotranx_launch(n_cells, 1, [&] { tor_ord::init_state_values(states.data(), n_cells); });
    for (long long c = 0; c < n_cells; ++c) activation[c] = -1;

    const double dt = cable::dt;
    const int steps = static_cast<int>(t_end / dt);
    for (int step = 0; step < steps; ++step) {
        const double t = step * dt;
        double* s = states.data();
        gotranx_launch(n_cells, 1, [&] {
            tor_ord::generalized_rush_larsen(s, t, dt, parameters.data(), s, n_cells);
        });
        const double stimulus = cable::stimulus(t);
        gotranx_launch(n_cells, 1, [&] {
            cable::diffuse(s + v * n_cells, v_new.data(), n_cells, stimulus);
        });
        for (long long c = 0; c < n_cells; ++c) {
            s[v * n_cells + c] = v_new[c];
            if (activation[c] < 0 && v_new[c] > 0) activation[c] = t + dt;
        }
    }
}
"""
    )
    compiler = cxx_compiler.with_flags(
        "-I", str(HERE), "-I", str(tmp_path), "-I", str(CABLE_DEMO), "-Wno-unused-function"
    )
    lib = ctypes.CDLL(str(compiler.shared_library(source)))
    lib.simulate.argtypes = [ctypes.c_longlong, ctypes.c_double, ctypes.c_void_p]

    n_cells = 40
    activation = numpy.zeros(n_cells)
    lib.simulate(n_cells, 30.0, activation.ctypes.data)
    assert (activation > 0).all(), activation
    assert (numpy.diff(activation) >= 0).all(), activation
    assert activation[-1] > activation[0]
