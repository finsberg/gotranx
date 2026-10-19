"""The helper the compile tests use (tests/compilers.py)."""

import shutil
import subprocess

import pytest
from compilers import find_c_compiler


def test_a_failed_compile_shows_the_compilers_diagnostics(c_compiler, tmp_path):
    source = tmp_path / "broken.c"
    source.write_text("int f(int a) { if (a = 1) return 2; return 3; }\n")
    with pytest.raises(subprocess.CalledProcessError) as info:
        c_compiler.shared_library(source)
    # Warnings are errors, and the reason must reach the test report
    assert "parentheses" in str(info.value)


def test_cc_with_arguments_is_split(c_compiler, monkeypatch, tmp_path):
    # e.g. CC="ccache gcc": the first word is the program, the rest its leading arguments.
    # `env <compiler>` stands in for a wrapper, with the compiler the other tests use
    if shutil.which("env") is None:
        pytest.skip("needs `env` on PATH")
    monkeypatch.setenv("CC", f"env {c_compiler.command[-1]}")
    compiler = find_c_compiler()
    assert compiler is not None
    assert compiler.command == (shutil.which("env"), c_compiler.command[-1])
    source = tmp_path / "ok.c"
    source.write_text("int f(void) { return 1; }\n")
    assert compiler.shared_library(source).is_file()


def test_a_failed_cpp_compile_shows_the_compilers_diagnostics(cxx_compiler, tmp_path):
    source = tmp_path / "broken.cpp"
    source.write_text("int f(int a) { if (a = 1) return 2; return 3; }\n")
    with pytest.raises(subprocess.CalledProcessError) as info:
        cxx_compiler.shared_library(source)
    assert "parentheses" in str(info.value)
