"""The helper the compile tests use (tests/compilers.py)."""

import subprocess

import pytest


def test_a_failed_compile_shows_the_compilers_diagnostics(c_compiler, tmp_path):
    source = tmp_path / "broken.c"
    source.write_text("int f(int a) { if (a = 1) return 2; return 3; }\n")
    with pytest.raises(subprocess.CalledProcessError) as info:
        c_compiler.shared_library(source)
    # Warnings are errors, and the reason must reach the test report
    assert "parentheses" in str(info.value)
