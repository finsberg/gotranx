"""The CUDA code of every model in `tests/odefiles`, compiled with nvcc.

nvcc compiles without a GPU, so these run in CI (`.github/workflows/cuda.yml`, CUDA 12 and
13) and wherever nvcc is installed (`conftest.nvcc`); the kernels' numerics are checked on
the CPU in `test_cuda_kernels.py`. Warnings are errors, and float code must not contain a
single double precision instruction. Run with `pytest -m cuda`.
"""

from pathlib import Path

import pytest

from gotranx.cli import gotran2cuda
from gotranx.codegen import CFormat
from gotranx.load import load_ode, ode_from_string
from gotranx.schemes import Scheme

pytestmark = pytest.mark.cuda

HERE = Path(__file__).parent
# ohara_rudy_cipa_v1_2017.ode is a git-ignored local file that does not load
# (DuplicateSymbolError), so it only exists in some checkouts. Skip it until it is fixed.
ODEFILES = sorted(
    path for path in (HERE / "odefiles").glob("*.ode") if path.name != "ohara_rudy_cipa_v1_2017.ode"
)
CARDIAC_MODEL = HERE / "odefiles" / "ToRORd_dyn_chloride.ode"


def compile_cuda(code: str, compiler, directory: Path) -> Path:
    (directory / "model.cuh").write_text(code)
    source = directory / "model.cu"
    source.write_text('#include "model.cuh"\n')
    return compiler.object_file(source)


@pytest.fixture(scope="module", params=ODEFILES, ids=lambda path: path.stem)
def ode(request):
    return load_ode(request.param)


@pytest.mark.parametrize("float_type", ["double", "float"])
def test_nvcc_compiles_every_model(ode, nvcc, cuda_arch, float_type, tmp_path):
    code = gotran2cuda.get_code(
        ode, scheme=list(Scheme), format=CFormat.none, float_type=float_type
    )
    compile_cuda(code, nvcc.with_flags(f"-arch={cuda_arch}"), tmp_path)


def test_nvcc_float_has_no_double_precision(ode, nvcc, cuda_arch, tmp_path):
    # ptxas reports any double precision instruction, which -Werror makes an error
    code = gotran2cuda.get_code(ode, scheme=list(Scheme), format=CFormat.none, float_type="float")
    strict = nvcc.with_flags(f"-arch={cuda_arch}", "-Xptxas", "--warn-on-double-precision-use")
    compile_cuda(code, strict, tmp_path)


@pytest.fixture(scope="module")
def tor_ord():
    return load_ode(CARDIAC_MODEL)


@pytest.mark.parametrize(
    "options",
    [
        {"scheme": list(Scheme), "remove_unused": True},
        {"scheme": [Scheme.hybrid_rush_larsen], "stiff_states": ["v", "cai"]},
        {"missing_values": {"INa": 0, "cai": 1}},
    ],
    ids=["remove_unused", "hybrid_stiff_states", "missing_values"],
)
def test_nvcc_options(tor_ord, nvcc, cuda_arch, options, tmp_path):
    code = gotran2cuda.get_code(tor_ord, format=CFormat.none, **options)
    compile_cuda(code, nvcc.with_flags(f"-arch={cuda_arch}"), tmp_path)


def test_nvcc_missing_variables(nvcc, cuda_arch, tmp_path):
    from test_missing_variables import TEXT

    full = ode_from_string(TEXT)
    ode = full - full.get_component("inputs")
    code = gotran2cuda.get_code(ode, scheme=list(Scheme), format=CFormat.none, namespace="split")
    compile_cuda(code, nvcc.with_flags(f"-arch={cuda_arch}"), tmp_path)


CABLE_DEMO = HERE / ".." / "examples" / "cuda-cable"


def test_nvcc_builds_the_cable_demo(nvcc, cuda_arch, tmp_path):
    ode = load_ode(CABLE_DEMO / "ToRORd_dyn_chloride.ode")
    code = gotran2cuda.get_code(
        ode, scheme=[Scheme.generalized_rush_larsen], format=CFormat.none, namespace="tor_ord"
    )
    (tmp_path / "tor_ord.cuh").write_text(code)
    for name in ("main.cu", "cable.cuh"):
        (tmp_path / name).write_text((CABLE_DEMO / name).read_text())
    # Links against the CUDA runtime, which needs no GPU either
    executable = nvcc.with_flags(f"-arch={cuda_arch}").executable(
        tmp_path / "main.cu", output=tmp_path / "cable"
    )
    assert executable.is_file()


def test_nvcc_links_a_header_included_from_two_translation_units(nvcc, cuda_arch, tmp_path):
    # A solver split over several .cu files must be able to include the model in each
    ode = load_ode(HERE / "odefiles" / "lorentz.ode")
    (tmp_path / "model.cuh").write_text(
        gotran2cuda.get_code(ode, scheme=list(Scheme), format=CFormat.none)
    )
    a = tmp_path / "a.cu"
    a.write_text(
        '#include "model.cuh"\nint b();\nint main() { return b() - lorentz::NUM_STATES; }\n'
    )
    b = tmp_path / "b.cu"
    b.write_text('#include "model.cuh"\nint b() { return lorentz::NUM_STATES; }\n')
    compiler = nvcc.with_flags(f"-arch={cuda_arch}")
    assert compiler.executable(a, b, output=tmp_path / "program").is_file()
    # and with separate compilation of device code
    rdc = compiler.with_flags("-rdc=true")
    assert rdc.executable(a, b, output=tmp_path / "program_rdc").is_file()
