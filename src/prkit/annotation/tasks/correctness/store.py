"""Discovery, scanning, and persistence for correctness annotation.

Operates on a directory of model-answer ``problem_*.json`` files (one per problem). A
file counts as *annotated* once it has an integer ``correctness`` field. Reads/writes go
through the shared helpers in :mod:`prkit.annotation.base` so all original fields survive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...base import list_problem_jsons, load_json, save_json

__all__ = [
    "list_problem_jsons",
    "load_json",
    "save_json",
    "has_correctness",
    "count_annotated",
    "all_files_annotated",
    "first_unannotated_index",
    "next_unannotated_index",
]


def has_correctness(data: dict[str, Any]) -> bool:
    """True if this problem already carries a ``correctness`` label."""
    return data.get("correctness") is not None


def count_annotated(paths: list[Path]) -> int:
    """Number of files that load and already have ``correctness``."""
    total = 0
    for path in paths:
        try:
            data = load_json(path)
        except (OSError, ValueError):
            continue
        if has_correctness(data):
            total += 1
    return total


def first_unannotated_index(paths: list[Path]) -> tuple[int, bool]:
    """Return ``(index_to_open, all_labelled)``.

    A file that fails to load is treated as unannotated so the annotator is taken to it.
    """
    for i, path in enumerate(paths):
        try:
            data = load_json(path)
        except (OSError, ValueError):
            return i, False
        if not has_correctness(data):
            return i, False
    return 0, bool(paths)


def all_files_annotated(paths: list[Path]) -> bool:
    """True if there is at least one file and every file has ``correctness``."""
    if not paths:
        return False
    for path in paths:
        try:
            data = load_json(path)
        except (OSError, ValueError):
            return False
        if not has_correctness(data):
            return False
    return True


def next_unannotated_index(paths: list[Path], start: int) -> int:
    """Smallest index ``>= start`` lacking ``correctness``; else the last index."""
    n = len(paths)
    if n == 0:
        return 0
    for j in range(max(0, start), n):
        try:
            data = load_json(paths[j])
        except (OSError, ValueError):
            return j
        if not has_correctness(data):
            return j
    return n - 1
