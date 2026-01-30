"""
Evaluation metrics for physical reasoning.

This package contains various metrics for evaluating model performance
on physical reasoning tasks.
"""

__version__ = "0.1.0"
__author__ = "Yinghuan Zhang"
__author_email__ = "yinghuan.flash@gmail.com"

from .base import BaseMetric
from .accuracy import AccuracyMetric

__all__ = [
    "BaseMetric",
    "AccuracyMetric",
]
