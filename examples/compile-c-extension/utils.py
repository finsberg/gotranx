import shutil
import subprocess as sp
from pathlib import Path

import numpy as np
from cmodel import CModel
import gotranx

HERE = Path(__file__).absolute().parent

MODEL_C_DIR = HERE


def cpath(odefile):
    return (MODEL_C_DIR / str(odefile)).with_suffix(".h")


def cbuild_dir(model):
    cfile = cpath(model)
    return MODEL_C_DIR.joinpath(f"build_{cfile.stem}")


def load_model(ode_file, rebuild=True, regenerate=False):
    # Check if ode_file is present
    cfile = cpath(ode_file)
    if not cfile.is_file() or regenerate:
        gotran2c(ode_file)
    if not cfile.is_file() or rebuild:
        build_c(ode_file)

    build_dir = cbuild_dir(ode_file)
    lib = np.ctypeslib.load_library(next(build_dir.joinpath("lib").iterdir()), HERE)

    ode = gotranx.load_ode(str(ode_file))
    return CModel(lib, ode)


def build_c(model):
    cfile = cpath(model)
    with open(MODEL_C_DIR.joinpath("template.c"), "r") as f:
        template = f.read()

    include_str = f'#include "{cfile.name}"\n'

    with open(MODEL_C_DIR.joinpath("demo.c"), "w") as f:
        f.write(include_str + template)

    model_name = cfile.stem

    build_dir = cbuild_dir(model)
    if build_dir.exists():
        shutil.rmtree(build_dir)

    build_dir.mkdir()

    sp.check_call(["cmake", f"-DCELL_LIBFILE={model_name}", ".."], cwd=build_dir)
    sp.check_call(["make"], cwd=build_dir)


def gotran2c(odefile):
    ode = gotranx.load_ode(odefile)
    # breakpoint()
    # Generate code and generalized rush larsen scheme
    code = gotranx.cli.gotran2c.get_code(
        ode,
        scheme=[
            gotranx.schemes.Scheme.forward_generalized_rush_larsen,
            gotranx.schemes.Scheme.forward_explicit_euler,
        ],
    )
    fname = Path(odefile).with_suffix(".h").name
    (MODEL_C_DIR / fname).write_text(code)
