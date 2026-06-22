"""Import-boundary guard: ``prkit.verify`` must stay light-import-clean.

Runs in a fresh subprocess (so the host test process's own imports cannot mask a
leak) and asserts that importing the facade — and exercising ``verify`` —
never pulls in provider SDKs, the dataset hub, the ``datasets`` library, or pandas.
This is the contract that makes ``prkit.verify`` a ``pip install``-and-call verifier.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Heavy/optional modules that must NOT be importable as a side effect of the
# verify path. We check ``google.genai`` (the provider SDK), never bare ``google``,
# which is a namespace-package ``.pth`` artifact present at interpreter start.
_FORBIDDEN = [
    "anthropic",
    "openai",
    "google.genai",
    "datasets",
    "pandas",
    "prkit.datasets",
    # The model-graded judge (pulls ``openai``) must stay off the verify path. We
    # forbid the ``llm_judge`` subpackage specifically — ``prkit.evaluation`` itself is
    # now legitimately reachable, since the pure EED/SEED algorithm core lives in
    # ``prkit.evaluation.edit_distance`` and the partial-credit scorer imports it.
    "prkit.evaluation.llm_judge",
    # SEED units. The vendored CMPhysBench baseline pulls ``pint`` only on its
    # unit-aware Numeric path; ``EedScorer``/``SeedScorer`` import the vendored core
    # lazily inside ``score()`` so ``pint`` stays off ``import prkit.scoring``/``verify``.
    # (``latex2sympy2_extended``/``antlr4`` are a required core dep already on this
    # path by design — deliberately NOT forbidden.)
    "pint",
]


def test_verify_path_does_not_import_heavy_deps():
    code = textwrap.dedent(f"""
        import sys
        import prkit.verify
        from prkit.verify import verify

        # Exercise the full lazy path: this triggers the SemanticsScorer/sympy
        # imports, which still must not drag in the forbidden modules.
        verify("3 m/s", "3 m/s")

        forbidden = {_FORBIDDEN!r}
        leaked = [name for name in forbidden if name in sys.modules]
        if leaked:
            print("LEAKED:" + ",".join(leaked))
            raise SystemExit(1)
        raise SystemExit(0)
        """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "heavy deps leaked into the prkit.verify import path:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
