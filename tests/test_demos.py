from pathlib import Path
import subprocess
import sys
import shutil
import pytest
import os
import platform

here = Path(__file__).parent


@pytest.fixture
def env():
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    return env


def examples():
    try:
        for f in (here / ".." / "examples").iterdir():
            if f.is_dir():
                yield f
    except FileNotFoundError:
        pass


@pytest.mark.example
@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@pytest.mark.parametrize(
    "example",
    (pytest.param(f, id=f.name) for f in examples()),
)
def test_examples(example, tmpdir, env):
    if not (example / "main.py").exists():
        return

    if "compile-c-extension" in example.name and platform.system() == "Windows":
        pytest.skip("Skipping compile-c-extension example on Windows")
        return

    shutil.copytree(example, tmpdir / example.name)

    ret = subprocess.run(
        [sys.executable, "main.py"], cwd=tmpdir / example.name, check=True, env=env
    )
    assert ret.returncode == 0
