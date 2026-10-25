import os
import shutil
import subprocess
import sys

import pytest
from compilers import Compiler, find_c_compiler, find_cxx_compiler, find_nvcc
from gotranx import Parser
from gotranx import TreeToODE
from gotranx.ode import make_ode


@pytest.fixture(scope="module")
def parser() -> Parser:
    return Parser(parser="lalr", debug=True, propagate_positions=True)


@pytest.fixture(scope="module")
def trans() -> TreeToODE:
    return TreeToODE()


@pytest.fixture(scope="module")
def ode_unused(parser, trans):
    expr = """
    parameters(a=0)
    parameters("My component",
    sigma=ScalarParam(12.0, description="Some description"),
    rho=21.0,
    beta=2.4,
    unused_parameter=1.0
    )
    states("My component", x=1.0, y=2.0,z=3.05, unused_state=1.0)

    expressions("My component")
    dunused_state_dt = 0
    unused_expression = 0
    dy_dt = x*(rho - z) - y # millivolt
    dx_dt = sigma*(-x + y)
    dz_dt = -beta*z + x*y
    """
    tree = parser.parse(expr)
    return make_ode(*trans.transform(tree))


def _unavailable(reason: str, env: str = "GOTRANX_REQUIRE_JULIA") -> None:
    """Skip, unless the environment variable `env` is set (to a non-empty value). CI sets
    GOTRANX_REQUIRE_JULIA in the Julia workflow and GOTRANX_REQUIRE_CC (C and C++) on Linux and
    macOS, so
    that a runner on which the tool is missing fails instead of passing on skips."""
    if os.environ.get(env):
        pytest.fail(f"{env} is set, but {reason}")
    pytest.skip(reason)


def _compiler(find, missing: str) -> Compiler:
    if sys.platform == "win32":
        pytest.skip("Compile tests do not run on Windows yet (MSVC support is deferred)")
    compiler = find()
    if compiler is None:
        _unavailable(missing, env="GOTRANX_REQUIRE_CC")
    assert compiler is not None  # _unavailable raises
    return compiler


@pytest.fixture(scope="session")
def c_compiler() -> Compiler:
    """A C compiler (`$CC`, cc, gcc or clang) for the tests that compile generated C."""
    return _compiler(find_c_compiler, "no C compiler found ($CC, cc, gcc, clang)")


@pytest.fixture(scope="session")
def nvcc() -> Compiler:
    """nvcc (`$NVCC`, on PATH, or from the pip package nvidia-cuda-nvcc), for the tests
    marked `cuda`. nvcc compiles without a GPU."""
    compiler = find_nvcc()
    if compiler is None:
        _unavailable(
            "nvcc not found ($NVCC, PATH or pip install nvidia-cuda-nvcc)",
            env="GOTRANX_REQUIRE_NVCC",
        )
    assert compiler is not None  # _unavailable raises
    return compiler


def pytest_generate_tests(metafunc):
    # The GPU architectures the `cuda` tests compile for, e.g. GOTRANX_CUDA_ARCHS=sm_70,sm_90
    if "cuda_arch" in metafunc.fixturenames:
        archs = os.environ.get("GOTRANX_CUDA_ARCHS", "sm_75,sm_90").split(",")
        metafunc.parametrize("cuda_arch", [arch.strip() for arch in archs if arch.strip()])


@pytest.fixture(scope="session")
def cxx_compiler() -> Compiler:
    """A C++ compiler (`$CXX`, c++, g++ or clang++) for the tests that compile generated C++."""
    return _compiler(find_cxx_compiler, "no C++ compiler found ($CXX, c++, g++, clang++)")


@pytest.fixture(scope="session")
def julia() -> str:
    """The `julia` executable on PATH."""
    executable = shutil.which("julia")
    if executable is None:
        _unavailable("julia is not on PATH")
    assert executable is not None  # _unavailable raises
    return executable


@pytest.fixture(scope="session")
def cytozoo(julia) -> str:
    """The `julia` executable, once CytoZoo.jl is known to load in its default environment."""
    result = subprocess.run(
        [julia, "--startup-file=no", "-e", "using CytoZoo"], capture_output=True, text=True
    )
    if result.returncode != 0:
        _unavailable(f"CytoZoo.jl does not load in julia's default environment:\n{result.stderr}")
    return julia


@pytest.fixture(scope="session")
def cytozoo_solver(julia) -> str:
    """The `julia` executable, once CytoZoo.jl and an ODE solver for it (OrdinaryDiffEqTsit5,
    SciMLBase) are known to load in its default environment."""
    result = subprocess.run(
        [julia, "--startup-file=no", "-e", "using CytoZoo, OrdinaryDiffEqTsit5, SciMLBase"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _unavailable(
            "CytoZoo.jl, OrdinaryDiffEqTsit5 and SciMLBase do not load in julia's default "
            f"environment:\n{result.stderr}"
        )
    return julia


@pytest.fixture(scope="session")
def cytozoo_rosenbrock(julia) -> str:
    """The `julia` executable, once CytoZoo.jl and the Rosenbrock solvers
    (OrdinaryDiffEqRosenbrock, ADTypes, SciMLBase) are known to load in its default
    environment."""
    result = subprocess.run(
        [
            julia,
            "--startup-file=no",
            "-e",
            "using CytoZoo, OrdinaryDiffEqRosenbrock, ADTypes, SciMLBase",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _unavailable(
            "CytoZoo.jl, OrdinaryDiffEqRosenbrock, ADTypes and SciMLBase do not load in "
            f"julia's default environment:\n{result.stderr}"
        )
    return julia


@pytest.fixture(scope="session")
def cytozoo_forwarddiff(julia) -> str:
    """The `julia` executable, once CytoZoo.jl and ForwardDiff are known to load in its default
    environment."""
    result = subprocess.run(
        [julia, "--startup-file=no", "-e", "using CytoZoo, ForwardDiff"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        _unavailable(
            "CytoZoo.jl and ForwardDiff do not load in julia's default environment:\n"
            f"{result.stderr}"
        )
    return julia


@pytest.fixture
def run_julia(julia, tmp_path):
    """Run a Julia script and return the lines it printed. A failing script fails the test,
    with Julia's error as the message."""

    def run(script: str) -> list:
        driver = tmp_path / "driver.jl"
        driver.write_text(script)
        result = subprocess.run(
            [julia, "--startup-file=no", str(driver)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip().splitlines()

    return run
