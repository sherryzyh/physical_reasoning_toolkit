"""
Pre-defined workflow combinations for common annotation tasks.

This module provides ready-to-use workflow configurations that combine
multiple workflow modules for specific use cases.
"""

from .domain_only_workflow import DomainOnlyWorkflow
from .plain_auto_workflow import PlainAutomaticWorkflow

__all__ = [
    "DomainOnlyWorkflow",
    "PlainAutomaticWorkflow"
]
