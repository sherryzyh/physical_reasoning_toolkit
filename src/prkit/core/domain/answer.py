"""
Answer models for physical reasoning evaluation.

This module provides a unified Answer class that handles all answer kinds
through composition rather than inheritance.
"""

from dataclasses import dataclass, field
from typing import Any

from .answer_kinds import AnswerObjectKind

AnswerValue = int | float | str


@dataclass
class Answer:
    """Unified answer class that handles all answer kinds through composition.

    ``answer_kind`` is the canonical :class:`AnswerObjectKind` (the toolkit-wide
    answer ontology). This is a coarse ingestion-time tag; the physics-semantics
    engine re-derives the precise ``object_kind`` independently when judging.
    """

    value: AnswerValue  # NUMBER: number; all other kinds: text
    answer_kind: AnswerObjectKind
    unit: str | None = None  # Used only for PHYSICAL_QUANTITY (e.g., "m/s²", "N")
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}

    def validate(self) -> bool:
        """Validate the answer based on its kind."""
        validators = {
            AnswerObjectKind.NUMBER: self._validate_number,
            AnswerObjectKind.PHYSICAL_QUANTITY: self._validate_string,
            AnswerObjectKind.EXPRESSION: self._validate_string,
            AnswerObjectKind.RELATION: self._validate_string,
            AnswerObjectKind.QUALITATIVE_LABEL: self._validate_string,
            AnswerObjectKind.BOOLEAN: self._validate_string,
            AnswerObjectKind.SIGN_DIRECTION: self._validate_string,
            AnswerObjectKind.DESCRIPTIVE_TEXT: self._validate_string,
            AnswerObjectKind.CHOICE: self._validate_option,
        }
        validator = validators.get(self.answer_kind)
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
        return self.answer_kind == AnswerObjectKind.NUMBER

    def is_equation(self) -> bool:
        """Check if this is an equation/relation answer."""
        return self.answer_kind == AnswerObjectKind.RELATION

    def is_physical_quantity(self) -> bool:
        """Check if this is a physical quantity (number + units) answer."""
        return self.answer_kind == AnswerObjectKind.PHYSICAL_QUANTITY

    def is_formula(self) -> bool:
        """Check if this is a formula/expression answer."""
        return self.answer_kind == AnswerObjectKind.EXPRESSION

    def is_text(self) -> bool:
        """Check if this is a free-form descriptive text answer."""
        return self.answer_kind == AnswerObjectKind.DESCRIPTIVE_TEXT

    def is_option(self) -> bool:
        """Check if this is an option/choice answer."""
        return self.answer_kind == AnswerObjectKind.CHOICE

    def is_numerical(self) -> bool:
        """Check if this has a numeric component (number or physical_quantity)."""
        return self.answer_kind in (
            AnswerObjectKind.NUMBER,
            AnswerObjectKind.PHYSICAL_QUANTITY,
        )

    def is_symbolic(self) -> bool:
        """Check if this is a symbolic/math answer (relation, expression, or physical_quantity)."""
        return self.answer_kind in (
            AnswerObjectKind.RELATION,
            AnswerObjectKind.EXPRESSION,
            AnswerObjectKind.PHYSICAL_QUANTITY,
        )

    # Numerical-specific methods
    def get_unit(self) -> str | None:
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
        return (
            isinstance(self.value, (int, float))
            and not isinstance(self.value, bool)
            and self.value > 0
        )

    def is_negative(self) -> bool:
        """Check if the numerical value is negative."""
        if not self.is_numerical():
            return False
        return (
            isinstance(self.value, (int, float))
            and not isinstance(self.value, bool)
            and self.value < 0
        )

    # Symbolic-specific methods
    def is_latex(self) -> bool:
        """Check if the symbolic answer contains LaTeX formatting."""
        if not self.is_symbolic():
            return False
        value = str(self.value)
        return "$" in value or "\\" in value

    def get_clean_expression(self) -> str:
        """Get the mathematical expression without LaTeX delimiters."""
        if not self.is_symbolic():
            return str(self.value)
        clean = str(self.value).strip()
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

    def contains_keywords(self, keywords: list[str]) -> bool:
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

    def get_option_index(self) -> int | None:
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
        return f"Answer(value={repr(self.value)}, answer_kind={self.answer_kind.value}, unit={repr(self.unit)})"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "value": self.value,
            "answer_kind": self.answer_kind.value,
        }
        if self.unit:
            result["unit"] = self.unit
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def get_value(self) -> AnswerValue:
        """Get the answer value."""
        return self.value

    def get_type(self) -> AnswerObjectKind:
        """Get the answer kind."""
        return self.answer_kind

    def get_type_name(self) -> str:
        """Get the answer kind as a string."""
        return self.answer_kind.value
