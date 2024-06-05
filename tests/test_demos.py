from pathlib import Path
import subprocess
import sys
import shutil
import pytest

here = Path(__file__).parent


@pytest.mark.skipif(sys.version_info < (3, 9), reason="requires python3.9 or higher")
@pytest.mark.parametrize(
    "example",
    (pytest.param(f, id=f.name) for f in (here / ".." / "examples").iterdir() if f.is_dir()),
)
def test_examples(example, tmpdir):
    shutil.copytree(example, tmpdir / example.name)
    ret = subprocess.run([sys.executable, "main.py"], cwd=tmpdir / example.name, check=True)
    assert ret.returncode == 0
