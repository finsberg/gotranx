"""Compilers for the tests that compile generated code.

`find_c_compiler` and `find_cxx_compiler` find a compiler, and `Compiler.shared_library`
builds a shared library a test can load with ctypes, with warnings as errors. Only gcc and
clang are supported; MSVC is deferred, so the compiler fixtures (conftest.py) skip on
Windows.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import os
import shlex
import shutil
import subprocess
from pathlib import Path

# monitor_values computes every intermediate, and a function need not use every argument
# (e.g. `t` in an autonomous model), so unused variables and parameters are expected
WARNINGS = ("-Wall", "-Wextra", "-Werror", "-Wno-unused-variable", "-Wno-unused-parameter")
C_FLAGS = ("-std=c99", *WARNINGS)
CXX_FLAGS = ("-std=c++17", *WARNINGS)
# nvcc: every warning is an error, except #177 "variable was declared but never referenced"
# (the unused variables above)
CUDA_FLAGS = ("-std=c++17", "-Werror", "all-warnings", "-diag-suppress", "177")


class CompileError(subprocess.CalledProcessError):
    """A failed compile, whose message includes the compiler's diagnostics."""

    def __str__(self) -> str:
        return "\n".join([super().__str__(), self.stdout or "", self.stderr or ""]).strip()


@dataclasses.dataclass(frozen=True)
class Compiler:
    #: The program and any leading arguments, e.g. ("/usr/bin/ccache", "gcc")
    command: tuple[str, ...]
    flags: tuple[str, ...]

    def with_flags(self, *extra: str) -> Compiler:
        return dataclasses.replace(self, flags=(*self.flags, *extra))

    def _run(self, args: list[str]) -> None:
        result = subprocess.run([*self.command, *self.flags, *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise CompileError(
                result.returncode, result.args, output=result.stdout, stderr=result.stderr
            )

    def object_file(self, source: Path) -> Path:
        """Compile `source` (without linking) to an object file next to it, and return its
        path. Raises `CompileError` like `shared_library`."""
        output = source.with_suffix(".o")
        self._run(["-c", "-o", str(output), str(source)])
        return output

    def shared_library(self, *sources: Path) -> Path:
        """Compile and link `sources` into a shared library next to the first one, and
        return its path.

        Raises `CompileError` (a `subprocess.CalledProcessError`), whose message includes
        the compiler's output, if it fails.
        """
        library = sources[0].with_suffix(".so")
        result = subprocess.run(
            [
                *self.command,
                *self.flags,
                "-shared",
                "-fPIC",
                "-o",
                str(library),
                *map(str, sources),
                "-lm",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise CompileError(
                result.returncode, result.args, output=result.stdout, stderr=result.stderr
            )
        return library


def _find(env: str, candidates: tuple[str, ...], flags: tuple[str, ...]) -> Compiler | None:
    """`$env` if it is set and its program is found, otherwise the first candidate on PATH."""
    words_list = (
        [shlex.split(os.environ[env])] if os.environ.get(env) else [[c] for c in candidates]
    )
    for words in words_list:
        program = shutil.which(words[0])
        if program is not None:
            return Compiler(command=(program, *words[1:]), flags=flags)
    return None


def find_c_compiler() -> Compiler | None:
    """`$CC` (which may carry arguments), otherwise the first of cc, gcc and clang."""
    return _find("CC", ("cc", "gcc", "clang"), C_FLAGS)


def find_cxx_compiler() -> Compiler | None:
    """`$CXX` (which may carry arguments), otherwise the first of c++, g++ and clang++."""
    return _find("CXX", ("c++", "g++", "clang++"), CXX_FLAGS)


def _pip_nvcc() -> str | None:
    """nvcc from the pip package nvidia-cuda-nvcc (nvidia/cu<major>/bin/nvcc), the newest."""
    spec = importlib.util.find_spec("nvidia")
    if spec is None or spec.submodule_search_locations is None:
        return None
    candidates = [
        path
        for location in spec.submodule_search_locations
        for path in Path(location).glob("cu*/bin/nvcc")
        if path.parent.parent.name[2:].isdigit()
    ]
    if not candidates:
        return None
    return str(max(candidates, key=lambda path: int(path.parent.parent.name[2:])))


def find_nvcc() -> Compiler | None:
    """`$NVCC` (which may carry arguments), otherwise nvcc on PATH, otherwise the nvcc of
    the pip package nvidia-cuda-nvcc."""
    found = _find("NVCC", ("nvcc",), CUDA_FLAGS)
    if found is not None or os.environ.get("NVCC"):
        return found
    program = _pip_nvcc()
    return None if program is None else Compiler(command=(program,), flags=CUDA_FLAGS)
