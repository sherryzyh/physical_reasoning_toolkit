"""
Workflow modules for physical annotation.

This package provides different workflow strategies for processing physics problems:
- WorkflowComposer: Modular workflow composition system
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .workflow_composer import WorkflowComposer

__all__ = [
    "WorkflowComposer",
]
