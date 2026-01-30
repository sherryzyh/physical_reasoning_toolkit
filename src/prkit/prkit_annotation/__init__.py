"""
PRKit Annotation Package

Annotation workflows and tools for physics problems in PRKit (physical-reasoning-toolkit).
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .workers import DomainLabeler, TheoremDetector
from .workflows import WorkflowComposer

__all__ = [
    "WorkflowComposer",
    "DomainLabeler",
    "TheoremDetector",
]
