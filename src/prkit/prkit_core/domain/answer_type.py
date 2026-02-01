"""
Answer type definitions for physical reasoning evaluation.

This module defines the different types of answers that can be compared
and their associated metadata for proper evaluation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class AnswerType(Enum):
    """Enumeration of supported answer types."""

    SYMBOLIC = "symbolic"
    NUMERICAL = "numerical"
    TEXTUAL = "textual"
    OPTION = "option"
