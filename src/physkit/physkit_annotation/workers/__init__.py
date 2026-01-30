"""
Annotators module for physical problem annotation.

This module contains individual annotator classes for each step of the
physical problem annotation process.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .domain_labeler import DomainLabeler
from .theorem_detector import TheoremDetector

__all__ = [
    "DomainLabeler",
    "TheoremDetector",
]
