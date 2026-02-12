"""
Answer models for physical reasoning evaluation.

This module provides a unified Answer class that handles all answer categories
through composition rather than inheritance.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .answer_category import AnswerCategory


@dataclass
class Answer:
    """Unified answer class that handles all answer categories through composition."""

    value: Any  # NUMBER: actual number; PHYSICAL_QUANTITY: numeric part; OPTION: option string; EQUATION/FORMULA/TEXT: plain string
    answer_category: AnswerCategory
    unit: Optional[str] = None  # Used only for PHYSICAL_QUANTITY (e.g., "m/s²", "N")
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def validate(self) -> bool:
        """Validate the answer based on its category."""
        validators = {
            AnswerCategory.NUMBER: self._validate_number,
            AnswerCategory.EQUATION: self._validate_string,
            AnswerCategory.PHYSICAL_QUANTITY: self._validate_string,
            AnswerCategory.FORMULA: self._validate_string,
            AnswerCategory.TEXT: self._validate_string,
            AnswerCategory.OPTION: self._validate_option,
        }
        validator = validators.get(self.answer_category)
        return validator() if validator else False

    def _validate_number(self) -> bool:
        """Validate number answers."""
        return isinstance(self.value, (int, float)) and not isinstance(self.value, bool)

    def _validate_string(self) -> bool:
        """Validate string-based answers (equation, formula, physical_quantity, text)."""
        return isinstance(self.value, str) and len(self.value.strip()) > 0

    def _validate_option(self) -> bool:
        """Validate option answers."""
        return isinstance(self.value, str) and len(self.value.strip()) > 0

    # Type checking methods
    def is_number(self) -> bool:
        """Check if this is a dimensionless number answer."""
        return self.answer_category == AnswerCategory.NUMBER

    def is_equation(self) -> bool:
        """Check if this is an equation answer."""
        return self.answer_category == AnswerCategory.EQUATION

    def is_physical_quantity(self) -> bool:
        """Check if this is a physical quantity (number + units) answer."""
        return self.answer_category == AnswerCategory.PHYSICAL_QUANTITY

    def is_formula(self) -> bool:
        """Check if this is a formula answer."""
        return self.answer_category == AnswerCategory.FORMULA

    def is_text(self) -> bool:
        """Check if this is a text answer."""
        return self.answer_category == AnswerCategory.TEXT

    def is_option(self) -> bool:
        """Check if this is an option answer."""
        return self.answer_category == AnswerCategory.OPTION

    def is_numerical(self) -> bool:
        """Check if this has a numeric component (number or physical_quantity)."""
        return self.answer_category in (
            AnswerCategory.NUMBER,
            AnswerCategory.PHYSICAL_QUANTITY,
        )

    def is_symbolic(self) -> bool:
        """Check if this is a symbolic/math answer (equation, formula, or physical_quantity)."""
        return self.answer_category in (
            AnswerCategory.EQUATION,
            AnswerCategory.FORMULA,
            AnswerCategory.PHYSICAL_QUANTITY,
        )

    # Numerical-specific methods
    def get_unit(self) -> Optional[str]:
        """Get the unit for numerical/physical quantity answers."""
        return self.unit if self.is_numerical() else None

    def has_unit(self) -> bool:
        """Check if the answer has a unit (physical quantity)."""
        return self.is_physical_quantity() or (
            self.is_number() and self.unit is not None
        )

    def is_integer(self) -> bool:
        """Check if the numerical value is an integer."""
        if not self.is_numerical():
            return False
        return isinstance(self.value, int) or (
            isinstance(self.value, float) and self.value.is_integer()
        )

    def is_positive(self) -> bool:
        """Check if the numerical value is positive."""
        if not self.is_numerical():
            return False
        return self.value > 0

    def is_negative(self) -> bool:
        """Check if the numerical value is negative."""
        if not self.is_numerical():
            return False
        return self.value < 0

    # Symbolic-specific methods
    def is_latex(self) -> bool:
        """Check if the symbolic answer contains LaTeX formatting."""
        if not self.is_symbolic():
            return False
        return "$" in self.value or "\\" in self.value

    def get_clean_expression(self) -> str:
        """Get the mathematical expression without LaTeX delimiters."""
        if not self.is_symbolic():
            return str(self.value)
        clean = self.value.strip()
        if clean.startswith("$$") and clean.endswith("$$"):
            clean = clean[2:-2].strip()
        elif clean.startswith("$") and clean.endswith("$"):
            clean = clean[1:-1].strip()
        return clean

    # Textual-specific methods
    def word_count(self) -> int:
        """Get the number of words in the text."""
        if not self.is_text():
            return 0
        return len(str(self.value).split())

    def char_count(self) -> int:
        """Get the number of characters in the text."""
        if not self.is_text():
            return 0
        return len(str(self.value))

    def is_short(self) -> bool:
        """Check if the text is short (less than 10 words)."""
        return self.word_count() < 10

    def is_long(self) -> bool:
        """Check if the text is long (more than 50 words)."""
        return self.word_count() > 50

    def contains_keywords(self, keywords: List[str]) -> bool:
        """Check if the text contains any of the specified keywords."""
        if not self.is_text():
            return False
        text_lower = str(self.value).lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    # Option-specific methods
    def is_letter_option(self) -> bool:
        """Check if the option is a letter (A, B, C, D, E)."""
        if not self.is_option():
            return False
        return str(self.value).upper() in ["A", "B", "C", "D", "E"]

    def is_yes_no(self) -> bool:
        """Check if the option is Yes/No."""
        if not self.is_option():
            return False
        return str(self.value).upper() in ["YES", "NO"]

    def is_true_false(self) -> bool:
        """Check if the option is True/False."""
        if not self.is_option():
            return False
        return str(self.value).upper() in ["TRUE", "FALSE"]

    def is_numeric_option(self) -> bool:
        """Check if the option is a number (1, 2, 3, 4, 5)."""
        if not self.is_option():
            return False
        return str(self.value) in ["1", "2", "3", "4", "5"]

    def get_option_index(self) -> Optional[int]:
        """Get the numeric index of the option if applicable."""
        if not self.is_option():
            return None
        value = str(self.value).upper()
        if self.is_letter_option():
            return ord(value) - ord("A")  # A=0, B=1, C=2, etc.
        elif self.is_numeric_option():
            return int(value) - 1  # 1=0, 2=1, 3=2, etc.
        return None

    # Utility methods
    def __str__(self) -> str:
        """String representation of the answer."""
        if self.is_numerical() and self.unit:
            return f"{self.value} {self.unit}"
        return str(self.value)

    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Answer(value={repr(self.value)}, answer_category={self.answer_category.value}, unit={repr(self.unit)})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {"value": self.value, "answer_category": self.answer_category.value}
        if self.unit:
            result["unit"] = self.unit
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def get_value(self) -> Any:
        """Get the answer value."""
        return self.value

    def get_type(self) -> AnswerCategory:
        """Get the answer category."""
        return self.answer_category

    def get_type_name(self) -> str:
        """Get the answer category as a string."""
        return self.answer_category.value
