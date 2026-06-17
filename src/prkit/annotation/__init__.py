"""Human-in-the-loop annotation tasks for physics problem datasets.

One orchestrator dispatches to named tasks:

- ``gold`` — an expert authors the gold label for a problem attribute (subtype ``domain``).
- ``correctness`` — a human judges model answers against the gold reference (Streamlit UI).

Run via the CLI (``prkit annotate <task> ...``) or programmatically::

    from prkit.annotation import run_annotation
    run_annotation("gold", subtype="domain", dataset="seephys", annotator="yh")
"""

from .base import AnnotationTask
from .orchestrator import AnnotationOrchestrator, run_annotation
from .tasks.correctness import CorrectnessAnnotationTask
from .tasks.gold import GoldAnnotationTask

__all__ = [
    "AnnotationTask",
    "AnnotationOrchestrator",
    "run_annotation",
    "GoldAnnotationTask",
    "CorrectnessAnnotationTask",
]
