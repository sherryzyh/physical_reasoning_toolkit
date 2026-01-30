"""
PhysKit Annotation Package

Annotation workflows and tools for physics problems.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .workflows import WorkflowComposer
from .workers import DomainLabeler, TheoremDetector

__all__ = [
    "WorkflowComposer",
    "DomainLabeler",
    "TheoremDetector",
]
