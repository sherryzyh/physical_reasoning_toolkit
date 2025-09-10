"""
Pre-defined workflow combinations for common annotation tasks.

This module provides ready-to-use workflow configurations that combine
multiple workflow modules for specific use cases.
"""

from .domain_only_workflow import DomainOnlyWorkflow
from .theorem_label_only_workflow import TheoremLabelOnlyWorkflow

__all__ = [
    "DomainOnlyWorkflow",
    "TheoremLabelOnlyWorkflow"
]
