"""
Answer type definitions for physical reasoning evaluation.

This module defines the different types of answers that can be compared
and their associated metadata for proper evaluation.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Union, Optional, Dict, List
from dataclasses import dataclass


class AnswerType(Enum):
    """Enumeration of supported answer types."""
    SYMBOLIC = "symbolic"
    NUMERICAL = "numerical"
    TEXTUAL = "textual"
    OPTION = "option"


@dataclass
class BaseAnswer(ABC):
    """Base class for all answer types with common functionality."""
    value: Any
    answer_type: AnswerType
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate the answer format."""
        pass
    
    def __str__(self) -> str:
        """String representation of the answer."""
        return str(self.value)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"{self.__class__.__name__}(value={repr(self.value)}, answer_type={self.answer_type.value})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert answer to dictionary for serialization."""
        return {
            "value": self.value,
            "answer_type": self.answer_type.value,
            "class": self.__class__.__name__
        }
    
    def get_value(self) -> Any:
        """Get the answer value."""
        return self.value
    
    def get_type(self) -> AnswerType:
        """Get the answer type."""
        return self.answer_type
    
    def get_type_name(self) -> str:
        """Get the answer type as a string."""
        return self.answer_type.value


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
    

    
    def is_latex(self) -> bool:
        """Check if the answer contains LaTeX formatting."""
        return '$' in self.value or '\\' in self.value
    
    def get_clean_expression(self) -> str:
        """Get the mathematical expression without LaTeX delimiters."""
        # Remove LaTeX delimiters if present
        clean = self.value.strip()
        if clean.startswith('$$') and clean.endswith('$$'):
            clean = clean[2:-2].strip()
        elif clean.startswith('$') and clean.endswith('$'):
            clean = clean[1:-1].strip()
        return clean


@dataclass
class NumericalAnswer(BaseAnswer):
    """Represents a numerical value with optional units and tolerance."""
    value: Union[int, float]
    unit: Optional[str] = None
    answer_type: AnswerType = AnswerType.NUMERICAL
    
    def validate(self) -> bool:
        """Validate that the value is a valid number."""
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)
    
    def __str__(self) -> str:
        """String representation of the answer."""
        return f"{self.value} {self.unit}" if self.unit else str(self.value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary including unit information."""
        result = super().to_dict()
        if self.unit:
            result["unit"] = self.unit
        return result
    
    def has_unit(self) -> bool:
        """Check if the answer has a unit."""
        return self.unit is not None
    
    def get_unit(self) -> Optional[str]:
        """Get the unit of the answer."""
        return self.unit
    
    def is_integer(self) -> bool:
        """Check if the value is an integer."""
        return isinstance(self.value, int) or (isinstance(self.value, float) and self.value.is_integer())
    
    def is_positive(self) -> bool:
        """Check if the value is positive."""
        return self.value > 0
    
    def is_negative(self) -> bool:
        """Check if the value is negative."""
        return self.value < 0


@dataclass
class TextualAnswer(BaseAnswer):
    """Represents a textual description or explanation."""
    value: str
    answer_type: AnswerType = AnswerType.TEXTUAL
    
    def validate(self) -> bool:
        """Validate that the value is a valid text."""
        return isinstance(self.value, str) and len(self.value.strip()) > 0
    

    
    def word_count(self) -> int:
        """Get the number of words in the text."""
        return len(self.value.split())
    
    def char_count(self) -> int:
        """Get the number of characters in the text."""
        return len(self.value)
    
    def is_short(self) -> bool:
        """Check if the text is short (less than 10 words)."""
        return self.word_count() < 10
    
    def is_long(self) -> bool:
        """Check if the text is long (more than 50 words)."""
        return self.word_count() > 50
    
    def contains_keywords(self, keywords: List[str]) -> bool:
        """Check if the text contains any of the specified keywords."""
        text_lower = self.value.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)


@dataclass
class OptionAnswer(BaseAnswer):
    """Represents a multiple choice answer."""
    value: str
    answer_type: AnswerType = AnswerType.OPTION
    
    def validate(self) -> bool:
        """Validate that the value is a valid option."""
        return isinstance(self.value, str) and len(self.value.strip()) > 0
    

    
    def is_letter_option(self) -> bool:
        """Check if the option is a letter (A, B, C, D, E)."""
        return self.value.upper() in ['A', 'B', 'C', 'D', 'E']
    
    def is_yes_no(self) -> bool:
        """Check if the option is Yes/No."""
        return self.value.upper() in ['YES', 'NO']
    
    def is_true_false(self) -> bool:
        """Check if the option is True/False."""
        return self.value.upper() in ['TRUE', 'FALSE']
    
    def is_numeric_option(self) -> bool:
        """Check if the option is a number (1, 2, 3, 4, 5)."""
        return self.value in ['1', '2', '3', '4', '5']
    
    def get_option_index(self) -> Optional[int]:
        """Get the numeric index of the option if applicable."""
        if self.is_letter_option():
            return ord(self.value.upper()) - ord('A')  # A=0, B=1, C=2, etc.
        elif self.is_numeric_option():
            return int(self.value) - 1  # 1=0, 2=1, 3=2, etc.
        return None

# Type alias for any answer type
Answer = Union[SymbolicAnswer, NumericalAnswer, TextualAnswer, OptionAnswer]
