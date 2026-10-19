"""Compilers for the tests that compile generated code.

`find_c_compiler` and `find_cxx_compiler` find a compiler, and `Compiler.shared_library`
builds a shared library a test can load with ctypes, with warnings as errors. Only gcc and
clang are supported; MSVC is deferred, so the compiler fixtures (conftest.py) skip on
Windows.
"""

from __future__ import annotations

import dataclasses
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
