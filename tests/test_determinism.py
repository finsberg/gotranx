"""Generated output must not depend on Python's per-process hash seed.

``sort_assignments`` feeds ``graphlib.TopologicalSorter``, which breaks ties in
insertion order. Any set of strings iterated on the way in therefore leaks
``PYTHONHASHSEED`` into the emitted order of states, monitors and intermediates,
and from there into the generated code.

The seed must be set before the interpreter starts, so every check here shells
out to a fresh interpreter rather than running in-process.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

here = Path(__file__).parent

# Models with enough independent branches for a tie-break to be observable.
# Small models have no ties to break, so they cannot expose this. These two produce a
# distinct order on every seed tried.
MODELS = ["ToRORd_dyn_chloride.ode", "ORdmm_Land.ode"]
SEEDS = ("1", "2", "7")

_ORDER_SCRIPT = """
import json, sys
import gotranx

ode = gotranx.load_ode(sys.argv[1])
# Marker-delimited: gotranx logs to stdout, so the payload cannot be the whole stream.
print("<<<ORDER>>>" + json.dumps({
    "assignments": [a.name for a in ode.sorted_assignments()],
    "states": [s.name for s in ode.sorted_states()],
}))
"""

_CODEGEN_SCRIPT = """
import sys
import gotranx

ode = gotranx.load_ode(sys.argv[1])
from gotranx.codegen import JuliaCodeGenerator
codegen = JuliaCodeGenerator(ode)
sys.stdout.write("<<<CODE>>>" + codegen.rhs() + codegen.monitor_index() + codegen.state_index())
"""

_CYTOZOO_SCRIPT = """
import sys
import gotranx

ode = gotranx.load_ode(sys.argv[1])
from gotranx.cli.gotran2cytozoo import get_code
sys.stdout.write("<<<CODE>>>" + get_code(ode))
"""


def _run(script: str, odefile: Path, seed: str, marker: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    out = subprocess.run(
        [sys.executable, "-c", script, str(odefile)],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert marker in out.stdout, f"marker missing; stderr was:\n{out.stderr}"
    return out.stdout.split(marker, 1)[1]


@pytest.mark.parametrize("model", MODELS)
def test_assignment_and_state_order_are_independent_of_hash_seed(model):
    odefile = here / "odefiles" / model
    orders = [json.loads(_run(_ORDER_SCRIPT, odefile, s, "<<<ORDER>>>")) for s in SEEDS]

    for seed, other in zip(SEEDS[1:], orders[1:]):
        assert other["assignments"] == orders[0]["assignments"], (
            f"assignment order under PYTHONHASHSEED={seed} differs from {SEEDS[0]}"
        )
        assert other["states"] == orders[0]["states"], (
            f"state order under PYTHONHASHSEED={seed} differs from {SEEDS[0]}"
        )


@pytest.mark.parametrize("model", MODELS)
def test_generated_code_is_independent_of_hash_seed(model):
    """The end-to-end property users actually depend on."""
    odefile = here / "odefiles" / model
    code = [_run(_CODEGEN_SCRIPT, odefile, s, "<<<CODE>>>") for s in SEEDS]

    for seed, other in zip(SEEDS[1:], code[1:]):
        assert other == code[0], (
            f"generated rhs under PYTHONHASHSEED={seed} differs from {SEEDS[0]}"
        )


@pytest.mark.parametrize("model", MODELS)
def test_cytozoo_code_is_independent_of_hash_seed(model):
    """The consuming repository regenerates and compares; this is what makes that safe.

    The CytoZoo backend adds orderings of its own on top of the Julia backend's
    -- the state, parameter and monitor name tuples, and the per-parameter
    `resolve_parameter` block -- so it needs its own end-to-end check rather
    than inheriting the one above.
    """
    odefile = here / "odefiles" / model
    code = [_run(_CYTOZOO_SCRIPT, odefile, s, "<<<CODE>>>") for s in SEEDS]

    for seed, other in zip(SEEDS[1:], code[1:]):
        assert other == code[0], (
            f"generated CytoZoo adapter under PYTHONHASHSEED={seed} differs from {SEEDS[0]}"
        )

    # And in-process: get_code is a pure function of its input, so loading the
    # same file twice must give the same bytes.
    from gotranx.cli.gotran2cytozoo import get_code
    from gotranx.load import load_ode

    assert get_code(load_ode(odefile)) == get_code(load_ode(odefile))
