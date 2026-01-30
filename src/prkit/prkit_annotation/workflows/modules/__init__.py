"""
Workflow modules for building composable annotation pipelines.

This module provides a set of workflow modules that can be chained or composed
together to quickly construct complex annotation workflows.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .detect_theorem_module import DetectTheoremModule
from .domain_assessment_module import DomainAssessmentModule
from .review_theorem_module import ReviewTheoremModule

__all__ = [
    "DomainAssessmentModule",
    "DetectTheoremModule",
    "ReviewTheoremModule",
]
