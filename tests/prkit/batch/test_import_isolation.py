"""Import-boundary guard: ``prkit.batch`` must stay a light leaf at module load.

Runs in a fresh subprocess (so the host test process's own imports cannot mask a
leak) and asserts that merely importing ``prkit.batch`` never pulls in a provider
SDK, the dataset hub, the semantics/api packages, ``pint``, ``pandas``, or even
``prkit.core.model_clients``. The submit orchestrator depends only on
``prkit.core.domain`` + stdlib at load; the client is duck-typed and the
``prkit.api`` version read is lazy (call-time, not import-time). The fetch half
(``fetch_batch`` / ``iter_batch_results``) imports ``batch_types`` lazily *inside*
its functions, so the new symbols are checked here for load-time cleanliness only —
a call-time no-leak assertion would fail by design (the live client already loaded
``prkit.core.model_clients``).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Heavy/optional modules that must NOT be importable as a side effect of merely
# importing the leaf. ``google.genai`` (the provider SDK) is checked, never bare
# ``google`` (a namespace-package ``.pth`` artifact present at interpreter start).
_FORBIDDEN = [
    "anthropic",
    "openai",
    "google.genai",
    "pint",
    "datasets",
    "pandas",
    "prkit.api",
    "prkit.datasets",
    "prkit.semantics",
    "prkit.core.model_clients",
]


def test_batch_import_does_not_pull_heavy_deps():
    code = textwrap.dedent(f"""
        import sys
        import prkit.batch
        from prkit.batch import (
            submit_batch_physics_reasoning,
            BatchSubmission,
            fetch_batch,
            iter_batch_results,
            batch_fetch_supported,
        )

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
        "heavy deps leaked into the prkit.batch import path:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
