"""
Answer type definitions for physical reasoning evaluation.

This module defines the different types of answers that can be compared
and their associated metadata for proper evaluation.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union, Optional
from dataclasses import dataclass


class AnswerType(Enum):
    """Enumeration of supported answer types."""
    SYMBOLIC = "symbolic"
    NUMERICAL = "numerical"
    TEXTUAL = "textual"


@dataclass
class BaseAnswer(ABC):
    """Base class for all answer types."""
    value: Any
    answer_type: AnswerType
    confidence: Optional[float] = None
    metadata: Optional[dict] = None
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the answer format."""
        pass
    
    def __str__(self) -> str:
        return f"{self.answer_type.value}: {self.value}"


@dataclass
class SymbolicAnswer(BaseAnswer):
    """Represents a symbolic mathematical expression or formula."""
    value: str  # Mathematical expression as string
    answer_type: AnswerType = AnswerType.SYMBOLIC
    
    def validate(self) -> bool:
        """Validate that the value is a valid symbolic expression."""
        # TODO: Implement symbolic expression validation
        # This could use sympy or similar library to check syntax
        return isinstance(self.value, str) and len(self.value.strip()) > 0


@dataclass
class NumericalAnswer(BaseAnswer):
    """Represents a numerical value with optional units and tolerance."""
    value: Union[int, float]
    units: Optional[str] = None
    tolerance: Optional[float] = None
    answer_type: AnswerType = AnswerType.NUMERICAL
    
    def validate(self) -> bool:
        """Validate that the value is a valid number."""
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)


@dataclass
class TextualAnswer(BaseAnswer):
    """Represents a textual description or explanation."""
    value: str
    answer_type: AnswerType = AnswerType.TEXTUAL
    
    def validate(self) -> bool:
        """Validate that the value is a valid text."""
        return isinstance(self.value, str) and len(self.value.strip()) > 0


# Type alias for any answer type
Answer = Union[SymbolicAnswer, NumericalAnswer, TextualAnswer]
