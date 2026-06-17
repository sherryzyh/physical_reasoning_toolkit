"""Single entry point that routes an annotation request to the right task.

The orchestrator holds a name -> :class:`AnnotationTask` registry and dispatches a run
to the matching task. This is the one place that knows about all available tasks, so a
caller (the CLI, a notebook, a script) only needs ``dispatch("gold", subtype="domain", ...)``.
"""

from __future__ import annotations

from typing import Any

from .base import AnnotationTask
from .tasks.correctness import CorrectnessAnnotationTask
from .tasks.gold import GoldAnnotationTask


class AnnotationOrchestrator:
    """Registry of annotation tasks plus a :meth:`dispatch` that runs one."""

    def __init__(self, tasks: dict[str, AnnotationTask] | None = None) -> None:
        if tasks is None:
            tasks = {}
            for task in (GoldAnnotationTask(), CorrectnessAnnotationTask()):
                tasks[task.name] = task
        self._tasks = tasks

    def register(self, task: AnnotationTask) -> None:
        """Add or replace a task in the registry."""
        self._tasks[task.name] = task

    def available_tasks(self) -> list[str]:
        """Return the registered task names, sorted."""
        return sorted(self._tasks)

    def get(self, task: str) -> AnnotationTask:
        """Return the task registered under *task* or raise ``ValueError``."""
        try:
            return self._tasks[task]
        except KeyError:
            raise ValueError(
                f"Unknown annotation task {task!r}. "
                f"Available: {', '.join(self.available_tasks())}."
            ) from None

    def dispatch(self, task: str, **kwargs: Any) -> int:
        """Run *task* with *kwargs* and return its exit code."""
        return self.get(task).run(**kwargs)


def run_annotation(task: str, **kwargs: Any) -> int:
    """Convenience wrapper: dispatch *task* through a default orchestrator."""
    return AnnotationOrchestrator().dispatch(task, **kwargs)
