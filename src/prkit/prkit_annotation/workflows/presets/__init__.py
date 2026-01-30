"""
Pre-defined workflow combinations for common annotation tasks.

This module provides ready-to-use workflow configurations that combine
multiple workflow modules for specific use cases.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .domain_only_workflow import DomainOnlyWorkflow
from .theorem_label_only_workflow import TheoremLabelOnlyWorkflow

__all__ = [
    "DomainOnlyWorkflow",
    "TheoremLabelOnlyWorkflow",
]
