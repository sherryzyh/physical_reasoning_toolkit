"""
Workflow modules for physical annotation.

This package provides different workflow strategies for processing physics problems:
- SupervisedAnnotationWorkflow: Supervised workflow with human review after each step
- AnnotationWorkflow: Simple annotation workflow that chains annotators step by step
"""

from .supervised_annotation_workflow import SupervisedAnnotationWorkflow
from .annotation_workflow import AnnotationWorkflow

__all__ = [
    "SupervisedAnnotationWorkflow",
    "AnnotationWorkflow"
]
