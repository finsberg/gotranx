"""Compilers for the tests that compile generated code.

`find_c_compiler` finds a C compiler and `Compiler.shared_library` builds a shared library a
test can load with ctypes, with warnings as errors. Only gcc and clang are supported; MSVC is
deferred, so the `c_compiler` fixture (conftest.py) skips on Windows.
"""

from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
from pathlib import Path

C_FLAGS = [
    "-std=c99",
    "-Wall",
    "-Wextra",
    "-Werror",
    # monitor_values computes every intermediate, and a function need not use every
    # argument (e.g. `t` in an autonomous model), so these two are expected
    "-Wno-unused-variable",
    "-Wno-unused-parameter",
]


class CompileError(subprocess.CalledProcessError):
    """A failed compile, whose message includes the compiler's diagnostics."""

    def __str__(self) -> str:
        return "\n".join([super().__str__(), self.stdout or "", self.stderr or ""]).strip()


@dataclasses.dataclass(frozen=True)
class Compiler:
    path: str

    def shared_library(self, source: Path) -> Path:
        """Compile `source` to a shared library next to it, and return its path.

        Raises `CompileError` (a `subprocess.CalledProcessError`), whose message includes
        the compiler's output, if it fails.
        """
        library = source.with_suffix(".so")
        result = subprocess.run(
            [self.path, *C_FLAGS, "-shared", "-fPIC", "-o", str(library), str(source), "-lm"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise CompileError(
                result.returncode, result.args, output=result.stdout, stderr=result.stderr
            )
        return library


def find_c_compiler() -> Compiler | None:
    """`$CC` if it is set (and found), otherwise the first of cc, gcc and clang on PATH."""
    candidates = [os.environ["CC"]] if os.environ.get("CC") else ["cc", "gcc", "clang"]
    for candidate in candidates:
        path = shutil.which(candidate)
        if path is not None:
            return Compiler(path)
    return None
