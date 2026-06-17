"""Tests for AnnotationOrchestrator registry and dispatch."""

from __future__ import annotations

import pytest

from prkit.annotation.base import AnnotationTask
from prkit.annotation.orchestrator import AnnotationOrchestrator, run_annotation


class _DummyTask(AnnotationTask):
    name = "dummy"

    def run(self, **kwargs):
        return kwargs.get("code", 0)


def test_default_registry_has_both_tasks():
    orchestrator = AnnotationOrchestrator()
    assert orchestrator.available_tasks() == ["correctness", "gold"]


def test_get_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown annotation task"):
        AnnotationOrchestrator().get("nope")


def test_register_and_dispatch_custom_task():
    orchestrator = AnnotationOrchestrator(tasks={})
    orchestrator.register(_DummyTask())
    assert orchestrator.available_tasks() == ["dummy"]
    assert orchestrator.dispatch("dummy", code=7) == 7


def test_run_annotation_wrapper_dispatches():
    # Unknown task surfaces as ValueError through the convenience wrapper too.
    with pytest.raises(ValueError):
        run_annotation("does-not-exist")
