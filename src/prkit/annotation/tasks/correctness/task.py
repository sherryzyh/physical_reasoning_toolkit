"""The ``correctness`` annotation task: launch the Streamlit review UI.

A human judges whether each model answer is correct against the gold reference. Streamlit
must own its own process, so this task shells out to ``streamlit run`` and forwards the
answers directory / annotator name to the app after ``--``.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar

from ...base import AnnotationTask

_UI_APP = Path(__file__).resolve().parent / "ui" / "app.py"

Runner = Callable[[Sequence[str]], int]


def _default_runner(command: Sequence[str]) -> int:
    return subprocess.run(list(command), check=False).returncode


class CorrectnessAnnotationTask(AnnotationTask):
    """Human correctness judgement over a directory of model answers."""

    name: ClassVar[str] = "correctness"

    def run(  # type: ignore[override]
        self,
        answers_dir: str | Path | None = None,
        *,
        annotator: str = "",
        port: int | None = None,
        no_browser: bool = False,
        runner: Runner | None = None,
        **_: Any,
    ) -> int:
        """Launch the Streamlit UI against *answers_dir* and return its exit code."""
        if not answers_dir:
            raise ValueError("correctness annotation needs an answers directory.")
        folder = Path(answers_dir).expanduser()
        if not folder.is_dir():
            raise ValueError(f"Not a directory: {folder}")

        if importlib.util.find_spec("streamlit") is None:
            print(
                "Streamlit is required for the correctness UI. Install it with:\n"
                '  pip install "physical-reasoning-toolkit[annotation]"',
                file=sys.stderr,
            )
            return 1

        command = self._build_command(folder, annotator, port, no_browser)
        run = runner or _default_runner
        return run(command)

    @staticmethod
    def _build_command(
        folder: Path, annotator: str, port: int | None, no_browser: bool
    ) -> list[str]:
        command = [sys.executable, "-m", "streamlit", "run", str(_UI_APP)]
        if port is not None:
            command += ["--server.port", str(port)]
        if no_browser:
            command += ["--server.headless", "true"]
        command += ["--", "--answers-dir", str(folder)]
        if annotator:
            command += ["--annotator", annotator]
        return command
