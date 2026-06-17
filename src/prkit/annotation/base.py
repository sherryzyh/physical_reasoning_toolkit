"""Shared building blocks for human-in-the-loop annotation tasks.

This module defines the :class:`AnnotationTask` contract that every task implements,
plus the small set of JSON helpers shared by the tasks (problem-file discovery,
loading, and an atomic save that preserves all existing fields).
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

_PROBLEM_RE = re.compile(r"problem_(\d+)\.json$", re.IGNORECASE)


class AnnotationTask(ABC):
    """Base class for an annotation task dispatched by :class:`AnnotationOrchestrator`.

    A task takes a dataset (or a directory of ``problem_*.json`` files) and drives a
    human through annotating it. Subclasses set :attr:`name` and implement :meth:`run`.
    """

    #: CLI/registry name for this task (e.g. ``"gold"`` or ``"correctness"``).
    name: ClassVar[str]

    @abstractmethod
    def run(self, **kwargs: Any) -> int:
        """Execute the task and return a process-style exit code (``0`` == success)."""
        raise NotImplementedError


def problem_json_sort_key(path: Path) -> tuple[int, str]:
    """Sort key that orders ``problem_<n>.json`` numerically, others last by name."""
    match = _PROBLEM_RE.search(path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (10**12, path.name)


def list_problem_jsons(folder: Path) -> list[Path]:
    """Return ``problem_*.json`` files in *folder*, numerically sorted."""
    paths = sorted(folder.glob("problem_*.json"), key=problem_json_sort_key)
    return [p for p in paths if p.is_file()]


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from *path* (UTF-8)."""
    with Path(path).open(encoding="utf-8") as handle:
        data: dict[str, Any] = json.load(handle)
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write *data* to *path* as indented UTF-8 JSON.

    The write goes to a sibling temp file and is then ``os.replace``-d into place, so a
    crash mid-write never leaves a truncated annotation file. All keys in *data* are
    preserved (the caller is expected to mutate a loaded dict, not a subset).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(tmp, path)
