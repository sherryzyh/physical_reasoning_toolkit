"""
Workflow modules for building composable annotation pipelines.

This module provides a set of workflow modules that can be chained or composed
together to quickly construct complex annotation workflows.
"""

from .base_module import BaseWorkflowModule
from .domain_assessment_module import DomainAssessmentModule

__all__ = [
    "BaseWorkflowModule",
    "DomainAssessmentModule"
]
